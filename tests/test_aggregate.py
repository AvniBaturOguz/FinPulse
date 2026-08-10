"""Deterministic tests for aggregate sentiment. No network, no model."""

import pytest

from finpulse.aggregate import aggregate_sentiment
from finpulse.config import BEARISH, BULLISH, NEUTRAL, SENTIMENT_LABELS
from finpulse.news import Article


def make(sentiments):
    """Build minimal articles carrying only the labels under test."""
    return [
        Article(ticker="AAPL", title=f"Headline {i}", sentiment=s, confidence=0.9)
        for i, s in enumerate(sentiments)
    ]


def test_empty_input_is_safe():
    summary = aggregate_sentiment([])
    assert summary.total == 0
    assert summary.is_empty
    assert summary.score == 0.0
    assert summary.dominant == NEUTRAL
    assert summary.counts == {label: 0 for label in SENTIMENT_LABELS}
    assert summary.percentages == {label: 0.0 for label in SENTIMENT_LABELS}


def test_none_input_is_safe():
    assert aggregate_sentiment(None).total == 0


def test_counts_and_percentages():
    summary = aggregate_sentiment(make([BULLISH, BULLISH, BEARISH, NEUTRAL]))
    assert summary.total == 4
    assert summary.counts == {BULLISH: 2, BEARISH: 1, NEUTRAL: 1}
    assert summary.percentages == {BULLISH: 50.0, BEARISH: 25.0, NEUTRAL: 25.0}
    assert summary.dominant == BULLISH
    assert summary.score == pytest.approx(0.25)


def test_percentages_sum_to_one_hundred_within_rounding():
    summary = aggregate_sentiment(make([BULLISH, BEARISH, NEUTRAL]))
    assert sum(summary.percentages.values()) == pytest.approx(100.0, abs=0.3)


@pytest.mark.parametrize(
    ("label", "expected_score"),
    [(BULLISH, 1.0), (BEARISH, -1.0), (NEUTRAL, 0.0)],
)
def test_single_class_extremes(label, expected_score):
    summary = aggregate_sentiment(make([label] * 5))
    assert summary.total == 5
    assert summary.dominant == label
    assert summary.score == pytest.approx(expected_score)
    assert summary.percentages[label] == 100.0


def test_balanced_bullish_and_bearish_report_neutral_dominant():
    summary = aggregate_sentiment(make([BULLISH, BULLISH, BEARISH, BEARISH]))
    assert summary.dominant == NEUTRAL
    assert summary.score == pytest.approx(0.0)


def test_three_way_tie_reports_neutral_dominant():
    assert aggregate_sentiment(make([BULLISH, BEARISH, NEUTRAL])).dominant == NEUTRAL


def test_neutral_headlines_dilute_the_score():
    mostly_neutral = aggregate_sentiment(make([BULLISH] + [NEUTRAL] * 9))
    all_bullish = aggregate_sentiment(make([BULLISH]))
    assert mostly_neutral.score == pytest.approx(0.1)
    assert all_bullish.score == pytest.approx(1.0)
    assert mostly_neutral.dominant == NEUTRAL


def test_score_stays_within_bounds():
    for sentiments in ([BULLISH] * 7, [BEARISH] * 7, [BULLISH, BEARISH] * 3):
        assert -1.0 <= aggregate_sentiment(make(sentiments)).score <= 1.0


def test_unclassified_articles_are_ignored():
    articles = make([BULLISH, BEARISH])
    articles.append(Article(ticker="AAPL", title="Never analyzed"))
    articles.append(Article(ticker="AAPL", title="Bogus label", sentiment="Sideways"))
    summary = aggregate_sentiment(articles)
    assert summary.total == 2
    assert summary.counts == {BULLISH: 1, BEARISH: 1, NEUTRAL: 0}
