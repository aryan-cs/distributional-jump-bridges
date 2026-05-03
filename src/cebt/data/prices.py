"""Public price retrieval and return utilities."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import requests


@dataclass(frozen=True)
class PriceBar:
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_url: str

    @classmethod
    def from_dict(cls, row: dict) -> PriceBar:
        return cls(
            ticker=str(row["ticker"]),
            date=date.fromisoformat(str(row["date"])),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            source_url=str(row.get("source_url", "")),
        )

    def to_dict(self) -> dict:
        row = asdict(self)
        row["date"] = self.date.isoformat()
        return row


def stooq_symbol(ticker: str) -> str:
    normalized = ticker.lower().replace("-", ".")
    return normalized if "." in normalized else f"{normalized}.us"


def stooq_url(ticker: str, start_date: str, end_date: str) -> str:
    d1 = start_date.replace("-", "")
    d2 = end_date.replace("-", "")
    return f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&d1={d1}&d2={d2}&i=d"


def fetch_stooq_prices(ticker: str, start_date: str, end_date: str) -> list[PriceBar]:
    url = stooq_url(ticker, start_date, end_date)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.lower().startswith("no data"):
        return []
    rows: list[PriceBar] = []
    for row in csv.DictReader(StringIO(text)):
        try:
            rows.append(
                PriceBar(
                    ticker=ticker,
                    date=date.fromisoformat(row["Date"]),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    source_url=url,
                )
            )
        except (KeyError, ValueError):
            continue
    return sorted(rows, key=lambda item: item.date)


def load_price_rows(path: str | Path) -> dict[str, list[PriceBar]]:
    grouped: dict[str, list[PriceBar]] = {}
    if not Path(path).exists():
        return grouped
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            bar = PriceBar.from_dict(json.loads(line))
            grouped.setdefault(bar.ticker, []).append(bar)
    return {ticker: sorted(bars, key=lambda item: item.date) for ticker, bars in grouped.items()}


def close_returns(bars: Iterable[PriceBar]) -> dict[date, float]:
    ordered = sorted(bars, key=lambda item: item.date)
    returns: dict[date, float] = {}
    previous = None
    for bar in ordered:
        if previous and previous.close > 0:
            returns[bar.date] = bar.close / previous.close - 1.0
        previous = bar
    return returns


def bar_by_date(bars: Iterable[PriceBar]) -> dict[date, PriceBar]:
    return {bar.date: bar for bar in bars}
