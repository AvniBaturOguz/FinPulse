# FinPulse

Financial news sentiment analysis for a stock ticker.

FinPulse pulls recent news for a symbol from Yahoo Finance, classifies each
headline with the pretrained [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
model, maps the result to **Bullish / Bearish / Neutral**, and presents the
per-article verdicts plus an aggregate view in a Streamlit dashboard.

No API keys, tokens or paid services are involved. Everything runs on CPU.

---

## Features

- Ticker input with normalization (`aapl` becomes `AAPL`), defaulting to `AAPL`
- Recent news retrieval through `yfinance`
- Defensive normalization: malformed records, missing publishers, missing URLs
  and unparseable timestamps are handled instead of crashing
- De-duplication by article id, URL or normalized title
- FinBERT headline classification with a confidence value per article
- Aggregate view: totals, per-class counts, percentages, dominant sentiment and
  a transparent net score
- Sentiment distribution chart and a linked, timestamped news list
- Model loaded once per session; results memoized with an explicit Refresh

---

## Architecture

```
FinPulse/
├── app.py                        # Streamlit dashboard (presentation only)
├── requirements.txt              # pinned direct dependencies
├── conftest.py                   # puts the project root on sys.path for pytest
├── finpulse/
│   ├── __init__.py
│   ├── config.py                 # model name, limits, label vocabulary
│   ├── news.py                   # yfinance access, normalization, dedupe, time parsing
│   ├── sentiment.py              # FinBERT loading, batched inference, label mapping
│   ├── aggregate.py              # counts, percentages, dominant sentiment, score
│   └── pipeline.py               # ticker -> analyzed articles + aggregates
├── scripts/
│   └── inspect_yfinance_news.py  # prints the live yfinance news schema
└── tests/
    ├── test_news.py              # offline normalization tests
    ├── test_aggregate.py         # offline aggregation tests
    ├── test_sentiment.py         # real FinBERT inference smoke tests
    └── test_pipeline_e2e.py      # offline assembly tests + live integration test
```

The layering is deliberate:

- `news.py` is the **only** module that imports `yfinance`, so a Yahoo schema
  change is contained in one file.
- `sentiment.py` is the **only** module that imports `torch`/`transformers`, and
  it knows nothing about finance APIs.
- `aggregate.py` is pure functions over already-classified articles.
- `pipeline.py` is the only module that joins all three.
- `app.py` is the only module that imports `streamlit`.

Because nothing inside the `finpulse` package imports Streamlit, the whole
analysis chain can be run and tested from a script or a test.

---

## Data flow

```
Ticker (user input)
   ↓  normalize_ticker: strip, uppercase, validate
yfinance  (Ticker(symbol).news)
   ↓  raw records
News normalization  (normalize_articles)
   ↓  Article objects: title, publisher, time, url, id  (deduped, newest first)
FinBERT tokenizer   (padding, truncation, max_length=64)
   ↓  input_ids + attention_mask
FinBERT model       (torch.inference_mode, eval)
   ↓  logits
softmax → probabilities → argmax
   ↓  class index + winning probability
Label mapping via model.config.id2label
   ↓  Bullish / Bearish / Neutral + confidence
Aggregate sentiment (counts, percentages, dominant, score)
   ↓
Streamlit dashboard
```

---

## Technology stack

| Package        | Version  | Why it is here                                            |
| -------------- | -------- | --------------------------------------------------------- |
| `yfinance`     | 1.5.2    | Free, keyless source of ticker news and quote metadata     |
| `transformers` | 5.15.0   | Loads the FinBERT tokenizer and classification head        |
| `torch`        | 2.13.0   | Inference backend (CPU wheel; no CUDA required)            |
| `pandas`       | 3.0.5    | Builds the small frame behind the distribution chart       |
| `streamlit`    | 1.61.1   | Dashboard UI, caching and widgets                          |
| `pytest`       | 9.1.1    | Test runner                                                |

Built and verified on **Python 3.12.10** (Windows 11).

---

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first analysis downloads the FinBERT weights (~440 MB, ~836 MB on disk with
the cache metadata) into the Hugging Face cache. This happens once; later runs
work from the cache. No token is required — the model is public.

---

## Running

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>). Enter a ticker
and press **Analyze**. **Refresh** discards the memoized result and re-fetches
from Yahoo.

Headless (no browser auto-open):

```powershell
python -m streamlit run app.py --server.headless true
```

Running the pipeline without the UI:

```powershell
python -c "from finpulse.pipeline import analyze_ticker; r = analyze_ticker('aapl'); print(r.status, r.summary.counts, r.summary.dominant, r.summary.score)"
```

Inspecting the live yfinance news schema (useful after upgrading yfinance):

```powershell
python scripts/inspect_yfinance_news.py AAPL
```

---

## How the sentiment pipeline works

**1. Retrieval.** `yf.Ticker(symbol).news` returns a list of records. On
yfinance 1.5.x each record nests the useful fields under `content`:

```
{ "id": "...", "content": { "title": ..., "pubDate": "2026-08-10T14:21:24Z",
    "provider": {"displayName": "Barrons.com"},
    "canonicalUrl": {"url": "https://..."}, "contentType": "STORY" } }
```

Older releases returned a flat record (`uuid`, `title`, `publisher`, `link`,
`providerPublishTime`). `news.py` reads both by trying a chain of candidate
field locations and taking the first non-empty value.

**2. Normalization.** Each record becomes an `Article`. Records that are not
dicts, or that have no usable headline, are skipped. Publication values are
parsed from Unix seconds, Unix milliseconds, ISO-8601 (including the trailing
`Z` form) or `datetime` objects; anything unparseable becomes `None` and
displays as `Unknown` rather than dropping the article. Articles are then
de-duplicated by id, URL or case-and-whitespace-normalized title, sorted newest
first, and capped.

**3. Tokenization.** Headlines are tokenized in batches with `padding=True`
(pad to the longest headline in the batch, not to a fixed 512) and
`truncation=True, max_length=64`. Headlines are short, so this keeps tensors
small; the attention mask makes padding harmless.

**4. Inference.** The model runs in `eval()` mode inside
`torch.inference_mode()`, producing `logits` of shape `(batch, 3)`.

**5. Logits to probabilities.** `torch.softmax(logits, dim=-1)` normalizes each
row into a probability distribution summing to 1. `.max(dim=-1)` returns the
winning probability and its class index in one pass.

**6. Label mapping.** The class index is resolved through
`model.config.id2label`, which for this checkpoint is
`{0: 'positive', 1: 'negative', 2: 'neutral'}` — note that positive comes
*first*, unlike most sentiment models. Mapping by index rather than by name
would silently swap Bullish and Bearish, so FinPulse always maps by name:

```
positive -> Bullish
negative -> Bearish
neutral  -> Neutral
```

**7. Confidence.** The winning class probability, in (0, 1]. With three classes
it can never fall below 1/3. It measures how decisively the model preferred one
class — not how likely the model is to be right.

**8. Aggregation.** Counts per label, percentages, and:

- **Dominant sentiment** — the most frequent label. An exact tie is reported as
  *Neutral*, because a feed split evenly between Bullish and Bearish is
  genuinely mixed.
- **Score** — `(bullish − bearish) / total`, in [−1.00, +1.00]. `+1.00` means
  every analyzed headline was Bullish, `−1.00` every one Bearish. Neutral
  headlines stay in the denominator, so heavy neutral coverage pulls the score
  toward zero. It is a plain balance of counts that can be checked by hand
  against the article list.

**9. Caching.** `finpulse.sentiment.load_finbert` is wrapped in `lru_cache`, and
`app.py` adds `st.cache_resource` on top, so the weights load once per session
instead of on every rerun — a cold load takes seconds while classifying a batch
of headlines takes milliseconds. Analysis results use `st.cache_data` with a
10-minute TTL, which the Refresh button clears.

---

## Important project files

| File                             | Responsibility                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| `app.py`                          | Dashboard layout, widgets, caches, status-driven rendering. No business logic.                |
| `finpulse/config.py`              | Model id, token/batch limits, default ticker, label vocabulary and the FinBERT label map.     |
| `finpulse/news.py`                | `normalize_ticker`, `fetch_raw_news`, `normalize_articles`, `parse_publish_time`, metadata.   |
| `finpulse/sentiment.py`           | `load_finbert`, `resolve_label_map`, `analyze_headlines` returning sentiment + confidence.    |
| `finpulse/aggregate.py`           | `aggregate_sentiment` producing counts, percentages, dominant label and score.                |
| `finpulse/pipeline.py`            | `analyze_ticker`: the whole chain, returning a status instead of raising.                     |
| `scripts/inspect_yfinance_news.py`| Prints a compact summary of the live news schema.                                             |
| `tests/`                          | Offline unit tests plus real model and real network integration tests.                        |

---

## Tests

```powershell
python -m pytest -q
```

- `test_news.py` and `test_aggregate.py` are fully offline and deterministic.
- `test_sentiment.py` loads the real FinBERT model and checks that probabilities
  are valid, that batched and single-item inference agree, and that only the
  three allowed labels are produced. It does not assert the semantic class of
  synthetic sentences, since that is a model judgement rather than a code
  contract.
- `test_pipeline_e2e.py` runs a live `AAPL` request through the entire chain. It
  **skips with a reason** when the network or news feed is unavailable rather
  than reporting a false pass.

No test requires an API key.

---

## Limitations

- `yfinance` is an unofficial client for Yahoo Finance, not an institutional
  low-latency news feed. Availability, coverage and response schema can change
  without notice, and requests may be throttled.
- Typically only ~10 recent articles are returned per ticker, so the aggregate
  describes a small, recent sample — not the full news cycle.
- Only the **headline** is classified. A headline can misrepresent the article
  it belongs to, and nuance in the body text is not considered.
- FinBERT can be wrong. It was trained on financial text of a particular era and
  can misread sarcasm, unusual phrasing or novel terminology.
- Confidence is the model's softmax probability, not a calibrated probability of
  being correct. A high confidence value is not proof of a correct label.
- The aggregate score weights every headline equally: no recency decay, no
  confidence weighting, no publisher credibility weighting.
- A ticker Yahoo does not cover is indistinguishable from a real ticker with no
  recent news; both produce an empty result.
- Company name and price are decorative extras from a separate endpoint. If that
  endpoint fails they are simply omitted, and news sentiment still works.
- Each running server process holds its own copy of the model in memory.

### Not investment advice

FinPulse analyzes the sentiment of financial news headlines. It does not predict
stock movements, does not constitute financial advice, is not a trading signal,
and is not a trading strategy. Do not make investment decisions based on it.
