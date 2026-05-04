"""Public Fama-French factor loading utilities."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import requests

from cebt.utils.io import write_jsonl

FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOMENTUM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


@dataclass(frozen=True)
class FactorRow:
    date: date
    mkt_rf: float
    smb: float
    hml: float
    rf: float
    mom: float | None = None
    source_url: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> FactorRow:
        mom_value = row.get("mom")
        return cls(
            date=date.fromisoformat(str(row["date"])),
            mkt_rf=float(row["mkt_rf"]),
            smb=float(row["smb"]),
            hml=float(row["hml"]),
            rf=float(row["rf"]),
            mom=None if mom_value in {None, ""} else float(mom_value),
            source_url=str(row.get("source_url", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["date"] = self.date.isoformat()
        return row


def fetch_french_daily_factors(
    output_path: str | Path,
    include_momentum: bool = True,
) -> list[FactorRow]:
    """Download daily Fama-French factors and cache decimal-return rows as JSONL."""

    ff3_text = _download_first_csv_from_zip(FF3_DAILY_URL)
    ff3 = _parse_ff3_daily(ff3_text, FF3_DAILY_URL)
    momentum: dict[date, float] = {}
    if include_momentum:
        momentum_text = _download_first_csv_from_zip(MOMENTUM_DAILY_URL)
        momentum = _parse_momentum_daily(momentum_text)

    rows = [
        FactorRow(
            date=day,
            mkt_rf=row.mkt_rf,
            smb=row.smb,
            hml=row.hml,
            rf=row.rf,
            mom=momentum.get(day),
            source_url=(
                FF3_DAILY_URL
                if not include_momentum
                else f"{FF3_DAILY_URL};{MOMENTUM_DAILY_URL}"
            ),
        )
        for day, row in sorted(ff3.items())
    ]
    write_jsonl(output_path, [row.to_dict() for row in rows])
    return rows


def load_factor_rows(path: str | Path) -> list[FactorRow]:
    """Load factor rows from the JSONL cache written by ``fetch_french_daily_factors``.

    A plain CSV with columns ``date,mkt_rf,smb,hml,rf[,mom]`` is also accepted. CSV values are
    interpreted as decimal daily returns; the raw Kenneth French ZIP files should be downloaded with
    ``fetch_french_daily_factors`` so the percent-to-decimal conversion is explicit.
    """

    factor_path = Path(path)
    if not factor_path.exists():
        return []
    text = factor_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "{":
        return [
            FactorRow.from_dict(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]
    rows: list[FactorRow] = []
    for row in csv.DictReader(StringIO(text)):
        rows.append(FactorRow.from_dict(row))
    return rows


def factor_rows_by_date(rows: list[FactorRow]) -> dict[date, FactorRow]:
    return {row.date: row for row in rows}


def _download_first_csv_from_zip(url: str) -> str:
    response = requests.get(url, timeout=90, headers={"User-Agent": "DJB research"})
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV file found in factor archive: {url}")
        with archive.open(names[0]) as handle:
            return handle.read().decode("utf-8", errors="replace")


def _parse_ff3_daily(text: str, source_url: str) -> dict[date, FactorRow]:
    rows: dict[date, FactorRow] = {}
    for cells in _data_rows(text):
        if len(cells) < 5:
            continue
        day = _parse_french_date(cells[0])
        rows[day] = FactorRow(
            date=day,
            mkt_rf=_percent_to_decimal(cells[1]),
            smb=_percent_to_decimal(cells[2]),
            hml=_percent_to_decimal(cells[3]),
            rf=_percent_to_decimal(cells[4]),
            source_url=source_url,
        )
    return rows


def _parse_momentum_daily(text: str) -> dict[date, float]:
    rows: dict[date, float] = {}
    for cells in _data_rows(text):
        if len(cells) < 2:
            continue
        rows[_parse_french_date(cells[0])] = _percent_to_decimal(cells[1])
    return rows


def _data_rows(text: str) -> list[list[str]]:
    rows = []
    started = False
    for cells in csv.reader(StringIO(text)):
        stripped = [cell.strip() for cell in cells]
        if not stripped or not stripped[0]:
            if started:
                break
            continue
        if stripped[0].isdigit() and len(stripped[0]) == 8:
            rows.append(stripped)
            started = True
        elif started:
            break
    return rows


def _parse_french_date(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def _percent_to_decimal(value: str) -> float:
    return float(value) / 100.0
