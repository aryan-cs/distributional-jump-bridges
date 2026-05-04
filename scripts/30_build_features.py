"""Build leakage-safe event-model tensor features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cebt.cli import load_run_config, output_dir, parse_args, resolve_path
from cebt.data.factors import fetch_french_daily_factors, load_factor_rows
from cebt.data.prices import PriceBar
from cebt.features.build import build_feature_bundle
from cebt.utils.io import read_jsonl


def main() -> None:
    args = parse_args("Build event-model features")
    config = load_run_config(args.config)
    out = output_dir(args, "data/processed/pilot")
    source_out = _feature_source_dir(config) or out
    events = read_jsonl(source_out / "events.jsonl")
    price_rows = read_jsonl(source_out / "prices.jsonl")
    if not events or not price_rows:
        raise SystemExit(f"Missing events.jsonl or prices.jsonl in {source_out}")
    prices_by_ticker = {}
    for row in price_rows:
        bar = PriceBar.from_dict(row)
        prices_by_ticker.setdefault(bar.ticker, []).append(bar)
    market_ticker = config.get("prices", {}).get("market_ticker", "SPY")
    market_bars = prices_by_ticker.get(market_ticker, [])
    if not market_bars:
        raise SystemExit(f"Missing market proxy price rows for {market_ticker}")
    factor_rows = _load_or_fetch_factors(config, out)
    bundle = build_feature_bundle(events, prices_by_ticker, market_bars, config, out, factor_rows)
    print(f"Wrote feature bundle with {bundle.x_pre.shape[0]} rows to {out / 'features.npz'}")


def _feature_source_dir(config: dict[str, Any]) -> Path | None:
    source = config.get("features", {}).get("source_processed_dir")
    return resolve_path(source) if source else None


def _load_or_fetch_factors(config: dict[str, Any], out: Path) -> list:
    labels = config.get("labels", {})
    mode = str(labels.get("mode", "spy")).lower()
    if mode not in {"ff3", "ff4"}:
        return []
    factor_path = labels.get("factor_path")
    factor_path = resolve_path(factor_path) if factor_path else out / "fama_french_daily.jsonl"
    rows = load_factor_rows(factor_path)
    if rows:
        return rows
    if not bool(labels.get("download_factors", True)):
        raise SystemExit(f"Missing Fama-French factor rows: {factor_path}")
    return fetch_french_daily_factors(
        factor_path,
        include_momentum=mode == "ff4",
    )


if __name__ == "__main__":
    main()
