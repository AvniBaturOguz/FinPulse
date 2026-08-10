"""End-to-end analysis: ticker -> news -> FinBERT -> aggregates.

This is the only module that joins the three layers, and it contains no UI
code, so the same call powers the Streamlit dashboard, a test, or a script.
Failures are returned as a status on the result rather than raised, because
every caller wants to display a message instead of a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from finpulse.aggregate import SentimentSummary, aggregate_sentiment
from finpulse.config import MAX_NEWS_ARTICLES
from finpulse.news import (
    Article,
    NewsFetchError,
    fetch_ticker_metadata,
    get_articles,
    normalize_ticker,
)
from finpulse.sentiment import analyze_headlines

#: Result statuses. Only OK carries analyzed articles.
STATUS_OK = "ok"
STATUS_NO_NEWS = "no_news"
STATUS_INVALID_TICKER = "invalid_ticker"
STATUS_ERROR = "error"


@dataclass
class AnalysisResult:
    """Everything the dashboard needs for one ticker request."""

    ticker: str
    status: str = STATUS_OK
    message: str = ""
    articles: list[Article] = field(default_factory=list)
    summary: SentimentSummary = field(default_factory=SentimentSummary)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def analyze_articles(articles: list[Article]) -> list[Article]:
    """Attach sentiment and confidence to each article, in place.

    Every headline goes through the model in one batched call rather than one
    call per article. ``analyze_headlines`` guarantees one result per input in
    input order, which is what makes this zip safe.
    """
    if not articles:
        return articles

    results = analyze_headlines([article.title for article in articles])
    for article, result in zip(articles, results):
        article.sentiment = result.sentiment
        article.confidence = result.confidence
    return articles


def analyze_ticker(
    raw_ticker: str,
    limit: int = MAX_NEWS_ARTICLES,
    include_metadata: bool = True,
) -> AnalysisResult:
    """Run the full pipeline for a user-supplied ticker.

    Steps: normalize the symbol, fetch and normalize news, classify every
    headline in batches, aggregate the labels, and optionally attach company
    metadata. Any failure short-circuits into a status the caller can render.
    """
    try:
        ticker = normalize_ticker(raw_ticker)
    except ValueError as exc:
        return AnalysisResult(
            ticker=str(raw_ticker).strip().upper(),
            status=STATUS_INVALID_TICKER,
            message=str(exc),
        )

    try:
        articles = get_articles(ticker, limit=limit)
    except NewsFetchError as exc:
        return AnalysisResult(ticker=ticker, status=STATUS_ERROR, message=str(exc))

    if not articles:
        return AnalysisResult(
            ticker=ticker,
            status=STATUS_NO_NEWS,
            message=(
                f"No recent news was returned for {ticker}. "
                "The symbol may be unrecognized or simply uncovered right now."
            ),
        )

    try:
        analyze_articles(articles)
    except Exception as exc:  # noqa: BLE001 - model/runtime failures are reportable
        return AnalysisResult(
            ticker=ticker,
            status=STATUS_ERROR,
            message=f"Sentiment analysis failed: {exc}",
            articles=articles,
        )

    return AnalysisResult(
        ticker=ticker,
        status=STATUS_OK,
        articles=articles,
        summary=aggregate_sentiment(articles),
        metadata=fetch_ticker_metadata(ticker) if include_metadata else {},
    )
