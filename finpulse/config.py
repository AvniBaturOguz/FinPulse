"""Shared configuration constants for FinPulse.

Everything that a human might reasonably want to tune (which model to use, how
many articles to analyze, how long a tokenized headline may be) is collected
here so that the functional modules stay free of magic numbers.
"""

# --- Model ---------------------------------------------------------------

#: Public Hugging Face model used for financial sentiment. Requires no token.
MODEL_NAME = "ProsusAI/finbert"

#: Headlines are short, so a small maximum sequence length keeps inference fast
#: without truncating meaningful text. Longer inputs are truncated.
MAX_TOKEN_LENGTH = 64

#: Number of headlines sent through the model in a single forward pass.
BATCH_SIZE = 16


# --- News ----------------------------------------------------------------

#: Ticker pre-filled in the dashboard input.
DEFAULT_TICKER = "AAPL"

#: Upper bound on how many normalized articles are analyzed per request.
MAX_NEWS_ARTICLES = 25

#: Shown when an article has no parseable publication timestamp.
UNKNOWN_TIME_DISPLAY = "Unknown"

#: Shown when an article has no publisher/source information.
UNKNOWN_PUBLISHER_DISPLAY = "Unknown"


# --- Sentiment labels ----------------------------------------------------

BULLISH = "Bullish"
BEARISH = "Bearish"
NEUTRAL = "Neutral"

#: The only sentiment values FinPulse ever produces, in display order.
SENTIMENT_LABELS = (BULLISH, BEARISH, NEUTRAL)

#: FinBERT's own class names (from ``model.config.id2label``) mapped to the
#: vocabulary FinPulse presents. Keys are compared lowercased so that label
#: casing differences between model revisions do not break the mapping.
FINBERT_LABEL_MAP = {
    "positive": BULLISH,
    "negative": BEARISH,
    "neutral": NEUTRAL,
}
