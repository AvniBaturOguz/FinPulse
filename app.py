"""FinPulse Streamlit dashboard.

Presentation only. Every piece of logic - retrieval, normalization, inference,
aggregation - lives in the ``finpulse`` package; this file decides what to show
and how, and holds the two Streamlit caches:

* ``st.cache_resource`` for the FinBERT weights, which must load once per
  session rather than on every widget interaction, and
* ``st.cache_data`` for analysis results, so re-rendering the page does not
  re-download news or re-run the model. The Refresh button clears it.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finpulse.config import (
    BEARISH,
    BULLISH,
    DEFAULT_TICKER,
    MAX_NEWS_ARTICLES,
    NEUTRAL,
    SENTIMENT_LABELS,
)
from finpulse.pipeline import (
    STATUS_INVALID_TICKER,
    STATUS_NO_NEWS,
    analyze_ticker,
)
from finpulse.sentiment import load_finbert

#: Streamlit's own colour markdown, so no raw HTML is needed for emphasis.
SENTIMENT_COLOURS = {BULLISH: "green", BEARISH: "red", NEUTRAL: "gray"}

CACHE_TTL_SECONDS = 600


st.set_page_config(page_title="FinPulse", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading the FinBERT model...")
def warm_model():
    """Load FinBERT once per Streamlit session.

    ``cache_resource`` is the right cache for objects that are expensive and
    unserializable: the model stays in memory and is shared across reruns,
    whereas ``cache_data`` would try to pickle several hundred megabytes.
    """
    return load_finbert()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_analysis(ticker: str, limit: int):
    """Run the pipeline, memoized per ticker for ten minutes."""
    return analyze_ticker(ticker, limit=limit)


def render_header():
    st.title("📈 FinPulse")
    st.caption("AI-powered financial news sentiment analysis")


def render_company(metadata: dict):
    """Optional context line. Absent metadata is simply not shown."""
    if not metadata:
        return
    name, price = metadata.get("name"), metadata.get("price")
    if not name and price is None:
        return

    left, right = st.columns([3, 1])
    left.subheader(name or metadata.get("ticker", ""))
    if price is not None:
        right.metric("Latest price", f"{price:,.2f} {metadata.get('currency') or ''}".strip())


def render_metrics(summary):
    total, counts = summary.total, summary.counts
    columns = st.columns(5)
    columns[0].metric("Total News", total)
    columns[1].metric("Bullish", counts[BULLISH], f"{summary.percentages[BULLISH]:.0f}%")
    columns[2].metric("Bearish", counts[BEARISH], f"{summary.percentages[BEARISH]:.0f}%")
    columns[3].metric("Neutral", counts[NEUTRAL], f"{summary.percentages[NEUTRAL]:.0f}%")
    columns[4].metric(
        "Dominant Sentiment",
        summary.dominant,
        help="Most frequent label. An exact tie is reported as Neutral.",
    )

    st.metric(
        "Sentiment score",
        f"{summary.score:+.2f}",
        help=(
            "(Bullish - Bearish) / total analyzed headlines. Ranges from -1.00 "
            "(every headline Bearish) to +1.00 (every headline Bullish). "
            "Neutral headlines count toward the total, pulling the score to 0."
        ),
    )


def render_distribution(summary):
    st.subheader("Sentiment distribution")
    frame = pd.DataFrame(
        {"Articles": [summary.counts[label] for label in SENTIMENT_LABELS]},
        index=list(SENTIMENT_LABELS),
    )
    st.bar_chart(frame, color="#4c78a8")


def render_articles(articles):
    st.subheader("Latest news")
    for article in articles:
        colour = SENTIMENT_COLOURS.get(article.sentiment, "gray")
        confidence = f"{article.confidence:.0%}" if article.confidence else "n/a"

        with st.container(border=True):
            headline, verdict = st.columns([5, 1])
            # Markdown link, not raw HTML - keeps the page free of unsafe HTML.
            headline.markdown(
                f"**[{article.title}]({article.url})**" if article.url
                else f"**{article.title}**"
            )
            headline.caption(f"{article.publisher} · {article.published_display}")
            verdict.markdown(f":{colour}[**{article.sentiment}**]")
            verdict.caption(f"confidence {confidence}")


def render_result(result):
    if result.status == STATUS_INVALID_TICKER:
        st.error(result.message)
        return
    if result.status == STATUS_NO_NEWS:
        st.warning(result.message)
        return
    if not result.ok:
        st.error(result.message or "Analysis failed.")
        return

    render_company(result.metadata)
    render_metrics(result.summary)
    render_distribution(result.summary)
    render_articles(result.articles)


def main():
    render_header()

    with st.form("ticker-form"):
        columns = st.columns([3, 1, 1])
        ticker = columns[0].text_input(
            "Ticker", value=DEFAULT_TICKER, placeholder="AAPL, NVDA, TSLA..."
        )
        # Submitting the form reruns the script; Analyze uses the cached
        # result, Refresh discards it first.
        columns[1].form_submit_button("Analyze", type="primary")
        refresh = columns[2].form_submit_button("Refresh")

    if refresh:
        # Drop the memoized result so the next run re-fetches from Yahoo.
        cached_analysis.clear()

    # The form keeps its value across reruns, so the default ticker is analyzed
    # on first load and the submit buttons simply re-trigger the same path.
    warm_model()
    with st.spinner(f"Analyzing news for {ticker.strip().upper()}..."):
        result = cached_analysis(ticker, MAX_NEWS_ARTICLES)
    render_result(result)

    st.divider()
    st.caption(
        "FinPulse analyzes the sentiment of financial news headlines. It is not "
        "investment advice, not a trading signal, and not a prediction of future "
        "price movement."
    )


main()
