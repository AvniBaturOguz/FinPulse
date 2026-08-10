"""Financial news retrieval (yfinance) and normalization.

This module is the only place that knows what Yahoo Finance news payloads look
like. Everything downstream works with :class:`Article` objects, so a future
yfinance schema change is contained here.

Observed schema, yfinance 1.5.2 (probed via ``scripts/inspect_yfinance_news.py``)::

    [
      {
        "id": "2261ac1a-...",
        "content": {
          "id": "2261ac1a-...",
          "contentType": "STORY" | "VIDEO",
          "title": "Airbnb CEO: Why AI is the best thing to happen to us",
          "description": "<p>...</p>",
          "summary": "...",
          "pubDate": "2026-08-10T10:00:00Z",
          "displayTime": "2026-08-10T10:00:00Z" | "",
          "provider": {"displayName": "Barrons.com", "url": ..., "sourceId": ...},
          "canonicalUrl": {"url": "https://...", "site": ..., "region": ..., "lang": ...},
          "clickThroughUrl": {"url": "https://..."} | None,
          "thumbnail": {...}, "finance": {...}
        }
      },
      ...
    ]

Older yfinance releases (0.2.x) returned a flat record instead::

    {"uuid": ..., "title": ..., "publisher": ..., "link": ...,
     "providerPublishTime": 1712345678}

Both shapes are accepted; unknown records are skipped rather than fatal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import yfinance as yf

from finpulse.config import (
    MAX_NEWS_ARTICLES,
    UNKNOWN_PUBLISHER_DISPLAY,
    UNKNOWN_TIME_DISPLAY,
)

#: Tickers are letters/digits plus the few separators Yahoo uses. The leading
#: character may also be "^", which prefixes index symbols such as ^GSPC.
_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$")

#: A headline shorter than this carries no usable sentiment signal.
_MIN_TITLE_LENGTH = 3

#: Timestamps above this are milliseconds, not seconds (year ~5138 in seconds).
_MILLISECOND_THRESHOLD = 100_000_000_000

_WHITESPACE = re.compile(r"\s+")


class NewsFetchError(RuntimeError):
    """Raised when the news request itself fails (network, invalid ticker)."""


@dataclass
class Article:
    """One normalized news item.

    ``sentiment`` and ``confidence`` are filled in later by the pipeline; this
    module never touches them, which keeps the news layer free of any ML
    dependency.
    """

    ticker: str
    title: str
    publisher: str = UNKNOWN_PUBLISHER_DISPLAY
    url: str | None = None
    published_at: datetime | None = None
    published_display: str = UNKNOWN_TIME_DISPLAY
    article_id: str | None = None
    content_type: str | None = None
    sentiment: str | None = None
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Flat dict form, convenient for pandas and the Streamlit layer."""
        return {
            "ticker": self.ticker,
            "title": self.title,
            "publisher": self.publisher,
            "url": self.url,
            "published_at": self.published_at,
            "published_display": self.published_display,
            "article_id": self.article_id,
            "content_type": self.content_type,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
        }


# --- ticker handling -----------------------------------------------------


def normalize_ticker(raw: str) -> str:
    """Normalize user input to Yahoo's ticker form (``" aapl "`` -> ``"AAPL"``).

    Raises ``ValueError`` for empty or clearly invalid input so that callers can
    show a message instead of firing a pointless network request.
    """
    if raw is None:
        raise ValueError("Ticker is required.")
    ticker = _WHITESPACE.sub("", str(raw)).upper()
    if not ticker:
        raise ValueError("Ticker is required.")
    if not _TICKER_PATTERN.match(ticker):
        raise ValueError(f"'{raw}' does not look like a valid ticker symbol.")
    return ticker


# --- time handling -------------------------------------------------------


def parse_publish_time(value: Any) -> datetime | None:
    """Best-effort conversion of a publication value into a UTC datetime.

    Accepts Unix seconds or milliseconds (int/float/numeric string), ISO-8601
    strings (including the trailing ``Z`` form yfinance 1.5 returns), and
    datetime objects. Anything unparseable yields ``None`` rather than raising,
    because a bad timestamp must never cost us an otherwise usable headline.
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, (int, float)):
        return _from_unix(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.replace(".", "", 1).isdigit():
            return _from_unix(float(text))
        # Python's parser accepts "+00:00" reliably across versions.
        iso = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    return None


def _from_unix(value: float) -> datetime | None:
    if abs(value) >= _MILLISECOND_THRESHOLD:
        value = value / 1000.0
    try:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def format_publish_time(moment: datetime | None) -> str:
    """Render a timestamp for display, or the ``Unknown`` fallback."""
    if moment is None:
        return UNKNOWN_TIME_DISPLAY
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --- field extraction ----------------------------------------------------


def _clean_str(value: Any) -> str | None:
    """Return a stripped non-empty string, or None for anything else."""
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _first_str(*candidates: Any) -> str | None:
    for candidate in candidates:
        text = _clean_str(candidate)
        if text:
            return text
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    """Nested fields are sometimes dicts, sometimes None, sometimes strings."""
    return value if isinstance(value, dict) else {}


def _extract_url(record: dict[str, Any], content: dict[str, Any]) -> str | None:
    """URLs live in nested dicts in 1.5.x and as a flat ``link`` in 0.2.x."""
    for key in ("clickThroughUrl", "canonicalUrl"):
        nested = _as_dict(content.get(key)) or _as_dict(record.get(key))
        url = _clean_str(nested.get("url"))
        if url:
            return url
        # Some responses collapse the dict into a bare string.
        url = _first_str(content.get(key), record.get(key))
        if url and url.startswith("http"):
            return url
    return _first_str(content.get("link"), record.get("link"), record.get("url"))


def _extract_publisher(record: dict[str, Any], content: dict[str, Any]) -> str:
    provider = _as_dict(content.get("provider")) or _as_dict(record.get("provider"))
    return (
        _first_str(
            provider.get("displayName"),
            provider.get("sourceId"),
            content.get("provider"),
            content.get("publisher"),
            record.get("publisher"),
        )
        or UNKNOWN_PUBLISHER_DISPLAY
    )


def _extract_publish_value(record: dict[str, Any], content: dict[str, Any]) -> Any:
    for source in (content, record):
        for key in ("pubDate", "displayTime", "providerPublishTime", "published_at"):
            value = source.get(key)
            if value not in (None, ""):
                return value
    return None


def _normalized_title_key(title: str) -> str:
    """Dedup key that ignores casing and whitespace differences."""
    return _WHITESPACE.sub(" ", title).strip().casefold()


def normalize_record(record: Any, ticker: str) -> Article | None:
    """Convert one raw yfinance record into an :class:`Article`.

    Returns ``None`` when the record is not a dict or carries no meaningful
    headline. Callers treat ``None`` as "skip this item".
    """
    if not isinstance(record, dict):
        return None

    # 1.5.x nests everything under "content"; 0.2.x is flat, so fall back to
    # the record itself and read both with the same code path.
    content = _as_dict(record.get("content"))

    title = _first_str(content.get("title"), record.get("title"), content.get("summary"))
    if not title or len(title) < _MIN_TITLE_LENGTH:
        return None

    published_at = parse_publish_time(_extract_publish_value(record, content))

    return Article(
        ticker=ticker,
        title=title,
        publisher=_extract_publisher(record, content),
        url=_extract_url(record, content),
        published_at=published_at,
        published_display=format_publish_time(published_at),
        article_id=_first_str(record.get("id"), content.get("id"), record.get("uuid")),
        content_type=_first_str(content.get("contentType"), record.get("type")),
    )


def normalize_articles(
    raw_records: Iterable[Any],
    ticker: str,
    limit: int = MAX_NEWS_ARTICLES,
) -> list[Article]:
    """Normalize, de-duplicate and sort raw records.

    De-duplication uses the first available of article id, URL or normalized
    title, so the same story arriving under two ids or two URLs is still
    collapsed. Results are ordered newest first, with undated items last.
    """
    articles: list[Article] = []
    seen: set[str] = set()

    for record in raw_records or []:
        try:
            article = normalize_record(record, ticker)
        except Exception:  # noqa: BLE001 - one bad record must not kill the batch
            continue
        if article is None:
            continue

        keys = [
            f"id:{article.article_id}" if article.article_id else None,
            f"url:{article.url}" if article.url else None,
            f"title:{_normalized_title_key(article.title)}",
        ]
        keys = [key for key in keys if key]
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        articles.append(article)

    articles.sort(
        key=lambda a: (a.published_at is not None, a.published_at or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    return articles[:limit]


# --- retrieval -----------------------------------------------------------


def fetch_raw_news(ticker: str) -> list[Any]:
    """Fetch the raw news payload for a ticker.

    Raises :class:`NewsFetchError` if yfinance cannot complete the request; an
    empty list means the request worked but Yahoo has no news for this symbol.
    """
    try:
        news = yf.Ticker(ticker).news
    except Exception as exc:  # noqa: BLE001 - network/parsing errors vary widely
        raise NewsFetchError(f"Could not retrieve news for {ticker}: {exc}") from exc

    if news is None:
        return []
    if not isinstance(news, list):
        raise NewsFetchError(
            f"Unexpected news payload for {ticker}: {type(news).__name__}"
        )
    return news


def get_articles(ticker: str, limit: int = MAX_NEWS_ARTICLES) -> list[Article]:
    """Fetch and normalize news for an already-normalized ticker."""
    return normalize_articles(fetch_raw_news(ticker), ticker, limit=limit)


def fetch_ticker_metadata(ticker: str) -> dict[str, Any]:
    """Best-effort company name and latest price. Never raises.

    This is decorative: the dashboard shows it when present and simply omits it
    otherwise, so a failure of Yahoo's quote endpoint can never take the news
    sentiment feature down with it. Each lookup is guarded separately because
    ``fast_info`` and ``info`` fail independently.
    """
    metadata: dict[str, Any] = {
        "ticker": ticker,
        "name": None,
        "price": None,
        "currency": None,
    }

    try:
        handle = yf.Ticker(ticker)
    except Exception:  # noqa: BLE001 - metadata must never break the request
        return metadata

    try:
        fast_info = handle.fast_info
        metadata["price"] = fast_info["last_price"]
        metadata["currency"] = fast_info["currency"]
    except Exception:  # noqa: BLE001
        pass

    try:
        info = handle.info or {}
        metadata["name"] = _first_str(info.get("shortName"), info.get("longName"))
        if metadata["price"] is None:
            metadata["price"] = info.get("regularMarketPrice")
        if metadata["currency"] is None:
            metadata["currency"] = info.get("currency")
    except Exception:  # noqa: BLE001
        pass

    return metadata
