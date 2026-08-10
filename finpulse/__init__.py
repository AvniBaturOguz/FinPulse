"""FinPulse - financial news sentiment analysis.

The package is split into layers that can be used independently:

* ``finpulse.config``    - shared constants and label mappings
* ``finpulse.news``      - yfinance retrieval and article normalization
* ``finpulse.sentiment`` - FinBERT loading and headline classification
* ``finpulse.aggregate`` - aggregate sentiment statistics
* ``finpulse.pipeline``  - ticker -> analyzed articles + aggregates

Nothing in this package imports Streamlit; the UI layer lives in ``app.py``.
"""

__version__ = "0.1.0"
