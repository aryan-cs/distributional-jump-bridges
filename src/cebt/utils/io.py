"""Small JSON/JSONL/CSV helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    value = Path(path)
    if not value.exists():
        return []
    rows: list[dict[str, Any]] = []
    with value.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")
    return output


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    materialized = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        output.write_text("", encoding="utf-8")
        return output
    fields = sorted({key for row in materialized for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    return output
