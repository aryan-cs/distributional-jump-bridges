"""Download public Stooq prices for event tickers and market proxy."""

from __future__ import annotations

from datetime import date, timedelta

from cebt.cli import load_run_config, output_dir, parse_args
from cebt.data.prices import fetch_stooq_prices
from cebt.utils.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    args = parse_args("Download prices")
    config = load_run_config(args.config)
    out = output_dir(args, "data/processed/pilot")
    events = read_jsonl(out / "events.jsonl")
    if not events:
        raise SystemExit(f"Missing events at {out / 'events.jsonl'}")
    data_config = config.get("data", {})
    pre_window = int(data_config.get("pre_window", 40))
    horizon = int(data_config.get("horizon", 5))
    start = (
        date.fromisoformat(data_config.get("start_date")) - timedelta(days=pre_window * 3)
    ).isoformat()
    end = (
        date.fromisoformat(data_config.get("end_date")) + timedelta(days=horizon * 3 + 10)
    ).isoformat()
    tickers = sorted(
        {event["ticker"] for event in events}
        | {config.get("prices", {}).get("market_ticker", "SPY")}
    )
    rows = []
    errors = []
    for ticker in tickers:
        try:
            rows.extend(bar.to_dict() for bar in fetch_stooq_prices(ticker, start, end))
        except Exception as exc:  # pragma: no cover - live network path
            errors.append({"ticker": ticker, "error": repr(exc)})
    write_jsonl(out / "prices.jsonl", rows)
    write_jsonl(out / "price_errors.jsonl", errors)
    write_json(
        out / "prices_summary.json",
        {"rows": len(rows), "tickers": len(tickers), "errors": len(errors)},
    )
    print(f"Wrote {len(rows)} price rows to {out / 'prices.jsonl'}")


if __name__ == "__main__":
    main()
