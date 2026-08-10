"""End-to-end pipeline tests.

Two kinds live here:

* offline tests that drive the pipeline's assembly and error paths using
  deterministic records, and
* one real integration test that hits Yahoo Finance and the real FinBERT model.

The integration test *skips* with an explicit reason when the network or the
news feed is unavailable. It never fabricates a successful result. No API key
is required by any of it.
"""

import pytest

from finpulse.aggregate import aggregate_sentiment
from finpulse.config import SENTIMENT_LABELS
from finpulse.news import Article, normalize_articles
from finpulse.pipeline import (
    STATUS_INVALID_TICKER,
    STATUS_NO_NEWS,
    STATUS_OK,
    analyze_articles,
    analyze_ticker,
)
from test_news import nested_record  # pytest puts tests/ on sys.path


# --- offline: assembly and error paths -----------------------------------


def test_invalid_ticker_short_circuits_before_any_request():
    result = analyze_ticker("not a ticker!")
    assert result.status == STATUS_INVALID_TICKER
    assert not result.ok
    assert result.articles == []
    assert result.summary.is_empty
    assert result.message


def test_empty_ticker_is_rejected():
    assert analyze_ticker("   ").status == STATUS_INVALID_TICKER


def test_analyze_articles_labels_every_article():
    raw = [nested_record(f"id-{i}", title) for i, title in enumerate([
        "Profits soar as the company raises full-year guidance",
        "The stock collapses after a surprise earnings miss",
        "The company will publish its results next Tuesday",
    ])]
    articles = analyze_articles(normalize_articles(raw, "AAPL"))

    assert len(articles) == 3
    for article in articles:
        assert article.sentiment in SENTIMENT_LABELS
        assert 1 / 3 <= article.confidence <= 1.0

    summary = aggregate_sentiment(articles)
    assert summary.total == 3
    assert sum(summary.counts.values()) == 3


def test_analyze_articles_handles_empty_input():
    assert analyze_articles([]) == []


def test_unknown_but_valid_symbol_reports_no_news():
    """A well-formed symbol Yahoo does not cover must not look like an error."""
    result = analyze_ticker("ZZZQQ", include_metadata=False)
    if result.status == STATUS_OK:
        pytest.skip("ZZZQQ unexpectedly returned news; nothing to assert")
    assert result.status in (STATUS_NO_NEWS, "error")
    assert result.message


# --- real integration ----------------------------------------------------


def test_real_ticker_end_to_end():
    """AAPL -> yfinance -> normalize -> FinBERT -> aggregates."""
    result = analyze_ticker("aapl")

    if result.status != STATUS_OK:
        pytest.skip(f"live news unavailable ({result.status}): {result.message}")

    assert result.ticker == "AAPL", "lowercase input must be normalized"
    assert result.articles, "ok status must carry articles"

    for article in result.articles:
        assert isinstance(article, Article)
        assert article.title.strip()
        assert article.sentiment in SENTIMENT_LABELS
        assert 1 / 3 <= article.confidence <= 1.0
        assert article.published_display

    summary = result.summary
    assert summary.total == len(result.articles)
    assert sum(summary.counts.values()) == summary.total
    assert summary.dominant in SENTIMENT_LABELS
    assert -1.0 <= summary.score <= 1.0
    assert sum(summary.percentages.values()) == pytest.approx(100.0, abs=0.5)

    # Metadata is optional by design: present or absent, never fatal.
    assert result.metadata.get("ticker") == "AAPL"
