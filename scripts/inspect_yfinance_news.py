"""Probe the shape of the news payload returned by the installed yfinance.

The yfinance news schema has changed between releases, so the parser in
``finpulse.news`` is written against what this script actually observes rather
than against a remembered structure. Re-run it after upgrading yfinance:

    python scripts/inspect_yfinance_news.py AAPL

It prints a compact structural summary (types, key names, truncated sample
values) and never dumps a full payload.
"""

from __future__ import annotations

import sys

import yfinance as yf

MAX_VALUE_CHARS = 70


def describe_value(value: object) -> str:
    """Return a short, type-aware description of a single field value."""
    if isinstance(value, dict):
        return f"dict(keys={sorted(value)[:8]})"
    if isinstance(value, (list, tuple)):
        inner = type(value[0]).__name__ if value else "empty"
        return f"{type(value).__name__}(len={len(value)}, first={inner})"
    text = repr(value)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + "..."
    return f"{type(value).__name__} = {text}"


def describe_record(record: object, indent: str = "  ") -> None:
    """Print one record's keys, one level of nesting deep."""
    if not isinstance(record, dict):
        print(f"{indent}(not a dict: {type(record).__name__})")
        return
    for key in record:
        value = record[key]
        print(f"{indent}{key}: {describe_value(value)}")
        if isinstance(value, dict):
            for sub_key in list(value)[:12]:
                print(f"{indent}    {sub_key}: {describe_value(value[sub_key])}")


def main(ticker: str) -> int:
    print(f"yfinance version: {yf.__version__}")
    print(f"ticker: {ticker}")

    try:
        news = yf.Ticker(ticker).news
    except Exception as exc:  # noqa: BLE001 - probe script reports any failure
        print(f"news request failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"top-level type: {type(news).__name__}")
    if not news:
        print("no news records returned")
        return 0

    print(f"record count: {len(news)}")
    print("\n--- record 0 ---")
    describe_record(news[0])

    if len(news) > 1:
        print("\n--- record 1 (key names only, to spot schema drift) ---")
        second = news[1]
        print(f"  keys: {sorted(second) if isinstance(second, dict) else type(second).__name__}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "AAPL"))
