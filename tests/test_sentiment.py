"""FinBERT inference smoke tests.

These exercise the real pretrained model, so the first run downloads it from
the Hugging Face Hub (public, no token). No API key is involved.

The assertions check that the *pipeline* is valid - shapes, ranges, ordering,
label vocabulary - and not that a given synthetic sentence must receive a
particular class, which is a model judgement rather than a code contract.
"""

import pytest
import torch

from finpulse.config import MAX_TOKEN_LENGTH, NEUTRAL, SENTIMENT_LABELS
from finpulse.sentiment import (
    SentimentResult,
    analyze_headlines,
    load_finbert,
    resolve_label_map,
)

HEADLINES = [
    "Company profits surge as revenue beats expectations",
    "Shares plunge after the firm slashes its full-year guidance",
    "The board will meet on Thursday to review the agenda",
]


@pytest.fixture(scope="module")
def finbert():
    return load_finbert()


def test_tokenizer_and_model_load(finbert):
    tokenizer, model = finbert
    assert tokenizer is not None and model is not None
    assert model.config.num_labels == 3
    assert not model.training, "model must be in eval mode"


def test_model_is_cached_not_reloaded(finbert):
    tokenizer, model = finbert
    tokenizer_again, model_again = load_finbert()
    assert model_again is model
    assert tokenizer_again is tokenizer


def test_label_map_covers_every_model_class(finbert):
    _, model = finbert
    label_map = resolve_label_map(model)
    assert set(label_map) == set(range(model.config.num_labels))
    assert set(label_map.values()) <= set(SENTIMENT_LABELS)
    # All three FinPulse labels should be reachable from the real checkpoint.
    assert set(label_map.values()) == set(SENTIMENT_LABELS)


def test_inference_returns_one_result_per_headline(finbert):
    results = analyze_headlines(HEADLINES)
    assert len(results) == len(HEADLINES)
    assert all(isinstance(result, SentimentResult) for result in results)


def test_sentiments_use_only_the_allowed_vocabulary(finbert):
    for result in analyze_headlines(HEADLINES):
        assert result.sentiment in SENTIMENT_LABELS


def test_confidence_is_a_valid_argmax_probability(finbert):
    for result in analyze_headlines(HEADLINES):
        # With three classes the winning probability cannot be below 1/3.
        assert 1 / 3 <= result.confidence <= 1.0


def test_probabilities_form_a_distribution(finbert):
    """Verify the logits -> softmax step directly, not just its output label."""
    tokenizer, model = finbert
    encoded = tokenizer(
        HEADLINES, padding=True, truncation=True,
        max_length=MAX_TOKEN_LENGTH, return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (len(HEADLINES), model.config.num_labels)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(len(HEADLINES)), atol=1e-5)

    # The reported confidence must be the maximum probability of its row.
    expected = probabilities.max(dim=-1).values.tolist()
    actual = [result.confidence for result in analyze_headlines(HEADLINES)]
    assert actual == pytest.approx(expected, abs=1e-5)


def test_batching_does_not_change_results(finbert):
    """A batch larger than batch_size must agree with one-at-a-time inference."""
    many = HEADLINES * 7  # 21 headlines, above the default batch size of 16
    batched = analyze_headlines(many, batch_size=8)
    single = analyze_headlines(many, batch_size=1)

    assert len(batched) == len(many)
    assert [r.sentiment for r in batched] == [r.sentiment for r in single]
    assert [r.confidence for r in batched] == pytest.approx(
        [r.confidence for r in single], abs=1e-4
    )


def test_empty_input_returns_empty_list():
    assert analyze_headlines([]) == []


def test_blank_headlines_keep_alignment(finbert):
    """Blanks are not classified but must not shift the other results."""
    mixed = [HEADLINES[0], "   ", HEADLINES[1]]
    results = analyze_headlines(mixed)
    reference = analyze_headlines([HEADLINES[0], HEADLINES[1]])

    assert len(results) == 3
    assert results[1] == SentimentResult(NEUTRAL, 0.0)
    assert results[0] == reference[0]
    assert results[2] == reference[1]


def test_long_headline_is_truncated_not_rejected(finbert):
    long_headline = "Quarterly earnings report " * 40
    results = analyze_headlines([long_headline])
    assert len(results) == 1
    assert results[0].sentiment in SENTIMENT_LABELS
