"""SEC event metadata and disclosure text ingestion."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from cebt.utils.hashing import sha256_text, stable_id
from cebt.utils.io import write_json
from cebt.utils.time import parse_datetime


def normalize_cik(cik: str | int, padded: bool = True) -> str:
    value = str(cik).strip()
    if value.upper().startswith("CIK"):
        value = value[3:]
    value = value.lstrip("0") or "0"
    return value.zfill(10) if padded else value


def normalize_accession(accession_number: str) -> str:
    return accession_number.replace("-", "")


@dataclass(frozen=True)
class Company:
    cik: str
    ticker: str
    name: str
    exchange: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["cik"] = normalize_cik(row["cik"])
        return row


@dataclass(frozen=True)
class DisclosureEvent:
    event_id: str
    ticker: str
    cik: str
    company_name: str
    accession_number: str
    form_type: str
    filing_date: str
    report_date: str | None
    accepted_at: str
    available_at: str
    primary_document: str
    source_url: str
    text_path: str | None = None
    text_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SECClient:
    def __init__(
        self,
        user_agent: str | None = None,
        base_url: str = "https://data.sec.gov",
        archives_url: str = "https://www.sec.gov/Archives/edgar/data",
        ticker_url: str = "https://www.sec.gov/files/company_tickers.json",
        max_requests_per_second: float = 8.0,
    ) -> None:
        self.user_agent = (
            user_agent or os.getenv("SEC_USER_AGENT") or "CEBT research contact@example.com"
        )
        self.base_url = base_url.rstrip("/")
        self.archives_url = archives_url.rstrip("/")
        self.ticker_url = ticker_url
        self.min_interval = 1.0 / max_requests_per_second if max_requests_per_second > 0 else 0.0
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html,*/*",
            }
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def get_json(self, url: str) -> dict[str, Any]:
        self._throttle()
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response.json()

    def get_text(self, url: str) -> str:
        self._throttle()
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response.text

    def company_tickers(self) -> list[Company]:
        payload = self.get_json(self.ticker_url)
        rows = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        fields = payload.get("fields", []) if isinstance(payload, dict) else []
        companies: list[Company] = []
        iterator = rows.values() if isinstance(rows, dict) else rows
        for item in iterator:
            if isinstance(item, dict):
                cik = item.get("cik_str") or item.get("cik") or item.get("CIK")
                ticker = item.get("ticker") or item.get("Ticker")
                name = item.get("title") or item.get("name") or item.get("Name")
                exchange = item.get("exchange")
            else:
                mapped = dict(zip(fields, item, strict=False))
                cik = mapped.get("cik")
                ticker = mapped.get("ticker")
                name = mapped.get("name")
                exchange = mapped.get("exchange")
            if cik and ticker and name:
                companies.append(Company(normalize_cik(cik), str(ticker), str(name), exchange))
        return companies

    def submissions_url(self, cik: str | int) -> str:
        return f"{self.base_url}/submissions/CIK{normalize_cik(cik)}.json"

    def get_submissions(self, cik: str | int, include_archives: bool = True) -> dict[str, Any]:
        payload = self.get_json(self.submissions_url(cik))
        if include_archives:
            archived = []
            for item in payload.get("filings", {}).get("files", []):
                name = item.get("name")
                if name:
                    archived.append(self.get_json(f"{self.base_url}/submissions/{name}"))
            payload["_archive_filings"] = archived
        return payload

    def document_url(self, cik: str, accession_number: str, primary_document: str) -> str:
        return (
            f"{self.archives_url}/{normalize_cik(cik, padded=False)}/"
            f"{normalize_accession(accession_number)}/{primary_document}"
        )

    def iter_events(
        self,
        submissions: dict[str, Any],
        company: Company,
        forms: set[str],
        start_date: str,
        end_date: str,
        max_events: int | None = None,
    ) -> Iterable[DisclosureEvent]:
        emitted = 0
        recent = submissions.get("filings", {}).get("recent", {})
        for payload in [recent, *submissions.get("_archive_filings", [])]:
            for event in self._iter_columnar(payload, company, forms, start_date, end_date):
                yield event
                emitted += 1
                if max_events is not None and emitted >= max_events:
                    return

    def _iter_columnar(
        self,
        payload: dict[str, list[Any]],
        company: Company,
        forms: set[str],
        start_date: str,
        end_date: str,
    ) -> Iterable[DisclosureEvent]:
        accessions = payload.get("accessionNumber", [])
        for idx, accession in enumerate(accessions):
            form_type = _column(payload, "form", idx)
            filing_date = _column(payload, "filingDate", idx)
            accepted = _column(payload, "acceptanceDateTime", idx)
            primary_document = _column(payload, "primaryDocument", idx)
            if (
                not accession
                or form_type not in forms
                or not filing_date
                or not accepted
                or not primary_document
            ):
                continue
            if filing_date < start_date or filing_date > end_date:
                continue
            accepted_at = parse_datetime(str(accepted)).isoformat()
            source_url = self.document_url(company.cik, str(accession), str(primary_document))
            yield DisclosureEvent(
                event_id=stable_id(company.cik, accession, form_type, prefix="event"),
                ticker=company.ticker,
                cik=normalize_cik(company.cik),
                company_name=company.name,
                accession_number=str(accession),
                form_type=str(form_type),
                filing_date=str(filing_date),
                report_date=_column(payload, "reportDate", idx),
                accepted_at=accepted_at,
                available_at=accepted_at,
                primary_document=str(primary_document),
                source_url=source_url,
            )

    def download_event_text(self, event: DisclosureEvent, raw_dir: str | Path) -> DisclosureEvent:
        target_dir = (
            Path(raw_dir)
            / "sec"
            / normalize_cik(event.cik, padded=False)
            / normalize_accession(event.accession_number)
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        html_path = target_dir / event.primary_document
        text_path = target_dir / "document.txt"
        metadata_path = target_dir / "event.json"
        if not html_path.exists():
            html_path.write_text(self.get_text(event.source_url), encoding="utf-8")
        html = html_path.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(html)
        text_path.write_text(text, encoding="utf-8")
        enriched = DisclosureEvent(
            **{**event.to_dict(), "text_path": str(text_path), "text_sha256": sha256_text(text)}
        )
        write_json(metadata_path, enriched.to_dict())
        return enriched


def select_companies(
    companies: list[Company],
    tickers: list[str],
    max_companies: int | None = None,
) -> list[Company]:
    by_ticker = {company.ticker.upper().replace(".", "-"): company for company in companies}
    if tickers:
        selected = [
            by_ticker[ticker.upper().replace(".", "-")]
            for ticker in tickers
            if ticker.upper().replace(".", "-") in by_ticker
        ]
    else:
        selected = sorted(companies, key=lambda row: (row.exchange or "", row.ticker))[
            : max_companies or 0
        ]
    return selected[:max_companies] if max_companies else selected


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def _column(payload: dict[str, list[Any]], key: str, idx: int) -> Any:
    values = payload.get(key, [])
    return None if idx >= len(values) else values[idx]
