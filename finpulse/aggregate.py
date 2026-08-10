"""Aggregate sentiment statistics over a set of analyzed articles.

Pure functions over plain data - no network, no model, no UI. Everything here
is a straightforward count, so the numbers shown on the dashboard can always be
traced back to the individual headlines that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from finpulse.config import BEARISH, BULLISH, NEUTRAL, SENTIMENT_LABELS


@dataclass
class SentimentSummary:
    """Aggregate view of one ticker's analyzed headlines."""

    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)
    dominant: str = NEUTRAL
    score: float = 0.0

    @property
    def is_empty(self) -> bool:
        """True when nothing was analyzed; callers should show a notice."""
        return self.total == 0


def aggregate_sentiment(articles: Iterable) -> SentimentSummary:
    """Count sentiments and derive percentages, dominant label and net score.

    Only articles carrying one of the three FinPulse labels are counted, so a
    record that was never classified cannot inflate the totals.

    ``dominant`` is the label with the highest count. A tie is reported as
    Neutral: if Bullish and Bearish arrive in equal numbers the coverage really
    is mixed, and picking either one would overstate the result.

    ``score`` is ``(bullish - bearish) / total`` and therefore lies in
    [-1.0, +1.0]. It is a plain balance of counts - +1.0 means every analyzed
    headline was Bullish, -1.0 means every one was Bearish, and 0.0 means the
    two cancel out (or everything was Neutral). Neutral headlines are included
    in the denominator, so heavy neutral coverage pulls the score toward zero.
    It is a description of news tone, not a prediction of price.
    """
    counts = {label: 0 for label in SENTIMENT_LABELS}
    for article in articles or []:
        sentiment = getattr(article, "sentiment", None)
        if sentiment in counts:
            counts[sentiment] += 1

    total = sum(counts.values())
    if total == 0:
        return SentimentSummary(
            total=0,
            counts=counts,
            percentages={label: 0.0 for label in SENTIMENT_LABELS},
            dominant=NEUTRAL,
            score=0.0,
        )

    percentages = {
        label: round(count * 100 / total, 1) for label, count in counts.items()
    }

    highest = max(counts.values())
    leaders = [label for label, count in counts.items() if count == highest]
    dominant = leaders[0] if len(leaders) == 1 else NEUTRAL

    score = round((counts[BULLISH] - counts[BEARISH]) / total, 3)

    return SentimentSummary(
        total=total,
        counts=counts,
        percentages=percentages,
        dominant=dominant,
        score=score,
    )
