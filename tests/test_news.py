"""Deterministic tests for news normalization.

No network access is required: fixtures mirror the real payload shapes observed
from yfinance 1.5.2 (nested ``content``) and yfinance 0.2.x (flat).
"""

from datetime import datetime, timezone

import pytest

from finpulse.config import UNKNOWN_PUBLISHER_DISPLAY, UNKNOWN_TIME_DISPLAY
from finpulse.news import (
    format_publish_time,
    normalize_articles,
    normalize_record,
    normalize_ticker,
    parse_publish_time,
)


def nested_record(
    article_id="id-1",
    title="Apple beats earnings expectations",
    publisher="Barrons.com",
    pub_date="2026-08-10T14:21:24Z",
    url=None,
    content_type="STORY",
):
    """Build a record in the shape yfinance 1.5.2 actually returns.

    The URL defaults to one derived from ``article_id`` so that distinct
    fixtures are genuinely distinct articles and do not trip de-duplication.
    """
    url = url or f"https://finance.yahoo.com/news/{article_id}.html"
    return {
        "id": article_id,
        "content": {
            "id": article_id,
            "contentType": content_type,
            "title": title,
            "summary": "Longer summary text.",
            "pubDate": pub_date,
            "displayTime": pub_date,
            "provider": {"displayName": publisher, "sourceId": "src"},
            "canonicalUrl": {"url": url, "site": "finance"},
            "clickThroughUrl": {"url": url},
            "thumbnail": {"resolutions": []},
        },
    }


def flat_record(article_id="legacy-1", title="Legacy headline about Apple"):
    """Build a record in the legacy yfinance 0.2.x shape."""
    return {
        "uuid": article_id,
        "title": title,
        "publisher": "Reuters",
        "link": "https://example.com/legacy",
        "providerPublishTime": 1_754_000_000,
        "type": "STORY",
    }


# --- ticker normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" msft ", "MSFT"), ("brk-b", "BRK-B"), ("^gspc", "^GSPC")],
)
def test_normalize_ticker_accepts_valid_input(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "not a ticker!", "A" * 20])
def test_normalize_ticker_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        normalize_ticker(raw)


# --- time parsing --------------------------------------------------------


def test_parse_iso_with_trailing_z():
    parsed = parse_publish_time("2026-08-10T14:21:24Z")
    assert parsed == datetime(2026, 8, 10, 14, 21, 24, tzinfo=timezone.utc)


def test_parse_unix_seconds_and_milliseconds_agree():
    seconds = parse_publish_time(1_754_000_000)
    millis = parse_publish_time(1_754_000_000_000)
    assert seconds == millis
    assert seconds.tzinfo is not None


def test_parse_naive_datetime_is_made_utc_aware():
    parsed = parse_publish_time(datetime(2026, 1, 2, 3, 4))
    assert parsed.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "value", [None, "", "   ", "yesterday", "not-a-date", True, {"x": 1}, []]
)
def test_unparseable_times_return_none_without_raising(value):
    assert parse_publish_time(value) is None


def test_format_publish_time_fallback():
    assert format_publish_time(None) == UNKNOWN_TIME_DISPLAY
    assert format_publish_time(
        datetime(2026, 8, 10, 14, 21, tzinfo=timezone.utc)
    ) == "2026-08-10 14:21 UTC"


# --- single record normalization -----------------------------------------


def test_nested_record_is_fully_normalized():
    article = normalize_record(nested_record(), "AAPL")
    assert article.ticker == "AAPL"
    assert article.title == "Apple beats earnings expectations"
    assert article.publisher == "Barrons.com"
    assert article.url == "https://finance.yahoo.com/news/id-1.html"
    assert article.article_id == "id-1"
    assert article.content_type == "STORY"
    assert article.published_at == datetime(2026, 8, 10, 14, 21, 24, tzinfo=timezone.utc)
    assert article.published_display == "2026-08-10 14:21 UTC"
    assert article.sentiment is None and article.confidence is None


def test_legacy_flat_record_is_supported():
    article = normalize_record(flat_record(), "AAPL")
    assert article.title == "Legacy headline about Apple"
    assert article.publisher == "Reuters"
    assert article.url == "https://example.com/legacy"
    assert article.article_id == "legacy-1"
    assert article.published_at is not None


@pytest.mark.parametrize(
    "record",
    [None, "a string", 42, [], {}, {"content": None}, {"content": {"title": "  "}}],
)
def test_records_without_usable_headline_are_skipped(record):
    assert normalize_record(record, "AAPL") is None


def test_missing_publisher_and_time_fall_back_gracefully():
    record = {"id": "x", "content": {"title": "A headline with no metadata"}}
    article = normalize_record(record, "AAPL")
    assert article.publisher == UNKNOWN_PUBLISHER_DISPLAY
    assert article.published_at is None
    assert article.published_display == UNKNOWN_TIME_DISPLAY
    assert article.url is None


def test_garbage_timestamp_does_not_lose_the_article():
    record = nested_record(pub_date="definitely not a date")
    article = normalize_record(record, "AAPL")
    assert article is not None
    assert article.published_display == UNKNOWN_TIME_DISPLAY


def test_unexpected_nested_types_are_tolerated():
    record = {
        "id": "weird",
        "content": {
            "title": "Nested fields have the wrong types",
            "provider": "a bare string, not a dict",
            "canonicalUrl": ["not", "a", "dict"],
            "clickThroughUrl": None,
        },
    }
    article = normalize_record(record, "AAPL")
    assert article.publisher == "a bare string, not a dict"
    assert article.url is None


# --- batch normalization -------------------------------------------------


def test_malformed_entries_do_not_break_the_batch():
    raw = [nested_record("a", "First real headline"), None, 7, {"junk": True},
           nested_record("b", "Second real headline")]
    articles = normalize_articles(raw, "AAPL")
    assert [a.title for a in articles] == ["First real headline", "Second real headline"]


def test_deduplication_by_id_url_and_title():
    raw = [
        nested_record("dup", "Headline one", url="https://a.example/1"),
        nested_record("dup", "Completely different text", url="https://b.example/2"),
        nested_record("other", "Headline two", url="https://a.example/1"),
        nested_record("third", "  HEADLINE   ONE  ", url="https://c.example/3"),
        nested_record("fourth", "Headline three", url="https://d.example/4"),
    ]
    articles = normalize_articles(raw, "AAPL")
    assert len(articles) == 2
    assert {a.article_id for a in articles} == {"dup", "fourth"}


def test_articles_are_sorted_newest_first_with_undated_last():
    raw = [
        nested_record("old", "Older story", pub_date="2026-08-01T00:00:00Z"),
        nested_record("none", "Undated story", pub_date=""),
        nested_record("new", "Newer story", pub_date="2026-08-09T00:00:00Z"),
    ]
    titles = [a.title for a in normalize_articles(raw, "AAPL")]
    assert titles == ["Newer story", "Older story", "Undated story"]


def test_limit_is_applied():
    raw = [nested_record(f"id-{i}", f"Headline number {i}") for i in range(10)]
    assert len(normalize_articles(raw, "AAPL", limit=4)) == 4


def test_empty_and_none_input_produce_no_articles():
    assert normalize_articles([], "AAPL") == []
    assert normalize_articles(None, "AAPL") == []


def test_as_dict_exposes_expected_keys():
    article = normalize_record(nested_record(), "AAPL")
    assert set(article.as_dict()) == {
        "ticker", "title", "publisher", "url", "published_at", "published_display",
        "article_id", "content_type", "sentiment", "confidence",
    }
