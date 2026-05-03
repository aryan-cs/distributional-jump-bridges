"""Build real SEC disclosure event rows."""

from __future__ import annotations

from cebt.cli import load_run_config, output_dir, parse_args, resolve_path, sec_client
from cebt.data.sec import select_companies
from cebt.utils.io import write_json, write_jsonl


def main() -> None:
    args = parse_args("Build SEC disclosure events")
    config = load_run_config(args.config)
    out = output_dir(args, "data/processed/pilot")
    raw_dir = resolve_path("data/raw")
    data_config = config.get("data", {})
    client = sec_client(config)
    companies = select_companies(
        client.company_tickers(),
        list(data_config.get("tickers", [])),
        data_config.get("max_companies"),
    )
    forms = set(data_config.get("forms", ["8-K"]))
    events = []
    errors = []
    for company in companies:
        try:
            submissions = client.get_submissions(company.cik, include_archives=True)
            for event in client.iter_events(
                submissions,
                company,
                forms=forms,
                start_date=data_config.get("start_date", "1900-01-01"),
                end_date=data_config.get("end_date", "2100-01-01"),
                max_events=data_config.get("max_events_per_company"),
            ):
                enriched = client.download_event_text(event, raw_dir)
                events.append(enriched.to_dict())
        except Exception as exc:  # pragma: no cover - live network path
            errors.append({"ticker": company.ticker, "cik": company.cik, "error": repr(exc)})
    write_jsonl(out / "events.jsonl", events)
    write_jsonl(out / "event_errors.jsonl", errors)
    write_json(
        out / "events_summary.json",
        {"events": len(events), "companies": len(companies), "errors": len(errors)},
    )
    print(f"Wrote {len(events)} events to {out / 'events.jsonl'}")


if __name__ == "__main__":
    main()
