"""FinBERT headline classification.

Inference only: the pretrained ``ProsusAI/finbert`` checkpoint is downloaded
from the Hugging Face Hub (public, no token required) and used as-is. Nothing
here is trained or fine-tuned.

Pipeline for a batch of headlines::

    headlines -> tokenizer(padding, truncation) -> model -> logits
              -> softmax -> class probabilities -> argmax
              -> id2label -> FinPulse label     (confidence = max probability)

This module deliberately does not import Streamlit, so it can be used from
tests and scripts. The UI adds its own ``st.cache_resource`` layer on top of
:func:`load_finbert`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finpulse.config import (
    BATCH_SIZE,
    FINBERT_LABEL_MAP,
    MAX_TOKEN_LENGTH,
    MODEL_NAME,
    NEUTRAL,
)


@dataclass(frozen=True)
class SentimentResult:
    """Classification of a single headline.

    ``confidence`` is the softmax probability of the winning class, so it lies
    in (0, 1]. With three classes it can never fall below 1/3.
    """

    sentiment: str
    confidence: float


@lru_cache(maxsize=1)
def load_finbert() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load the tokenizer and model once per process.

    The ``lru_cache`` is what stops a reload per request: the weights are a few
    hundred megabytes and loading them takes seconds, while classifying a batch
    of headlines takes milliseconds. ``model.eval()`` disables dropout so the
    same headline always yields the same result.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def resolve_label_map(model) -> dict[int, str]:
    """Map model class indices to FinPulse labels via ``model.config.id2label``.

    FinBERT reports ``{0: 'positive', 1: 'negative', 2: 'neutral'}`` - note that
    this is *not* the alphabetical ordering most sentiment checkpoints use, so
    hardcoding indices would silently mislabel everything. Reading the names
    from the config keeps the mapping correct regardless of ordering. Unknown
    class names fall back to Neutral rather than crashing.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    return {
        int(index): FINBERT_LABEL_MAP.get(str(name).strip().lower(), NEUTRAL)
        for index, name in id2label.items()
    }


def analyze_headlines(
    headlines: list[str],
    batch_size: int = BATCH_SIZE,
) -> list[SentimentResult]:
    """Classify headlines, returning one result per input in the same order.

    The 1:1 length guarantee matters because callers zip the results back onto
    their articles. Blank entries are not sent to the model; they get a Neutral
    placeholder so alignment is preserved.
    """
    if not headlines:
        return []

    tokenizer, model = load_finbert()
    label_map = resolve_label_map(model)

    results: list[SentimentResult | None] = [None] * len(headlines)
    usable = [(i, text.strip()) for i, text in enumerate(headlines) if str(text).strip()]

    for start in range(0, len(usable), batch_size):
        chunk = usable[start : start + batch_size]
        encoded = tokenizer(
            [text for _, text in chunk],
            padding=True,               # pad to the longest headline in the batch
            truncation=True,            # never exceed the model's limit
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )

        with torch.inference_mode():   # no autograd graph, no gradient memory
            logits = model(**encoded).logits

        # Raw logits are unbounded scores; softmax turns each row into a
        # probability distribution over the three classes.
        probabilities = torch.softmax(logits, dim=-1)
        confidences, class_indices = probabilities.max(dim=-1)

        for (index, _), class_index, confidence in zip(
            chunk, class_indices.tolist(), confidences.tolist()
        ):
            results[index] = SentimentResult(
                sentiment=label_map.get(class_index, NEUTRAL),
                confidence=float(confidence),
            )

    return [result or SentimentResult(NEUTRAL, 0.0) for result in results]
