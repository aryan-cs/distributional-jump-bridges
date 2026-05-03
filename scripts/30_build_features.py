"""Build leakage-safe CEBT tensor features."""

from __future__ import annotations

from cebt.cli import load_run_config, output_dir, parse_args
from cebt.data.prices import PriceBar
from cebt.features.build import build_feature_bundle
from cebt.utils.io import read_jsonl


def main() -> None:
    args = parse_args("Build CEBT features")
    config = load_run_config(args.config)
    out = output_dir(args, "data/processed/pilot")
    events = read_jsonl(out / "events.jsonl")
    price_rows = read_jsonl(out / "prices.jsonl")
    if not events or not price_rows:
        raise SystemExit("Missing events.jsonl or prices.jsonl")
    prices_by_ticker = {}
    for row in price_rows:
        bar = PriceBar.from_dict(row)
        prices_by_ticker.setdefault(bar.ticker, []).append(bar)
    market_ticker = config.get("prices", {}).get("market_ticker", "SPY")
    market_bars = prices_by_ticker.get(market_ticker, [])
    if not market_bars:
        raise SystemExit(f"Missing market proxy price rows for {market_ticker}")
    bundle = build_feature_bundle(events, prices_by_ticker, market_bars, config, out)
    print(f"Wrote feature bundle with {bundle.x_pre.shape[0]} rows to {out / 'features.npz'}")


if __name__ == "__main__":
    main()
