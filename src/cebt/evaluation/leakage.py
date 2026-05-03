"""Leakage validation for feature bundles."""

from __future__ import annotations

from pathlib import Path

from cebt.features.build import validate_feature_rows
from cebt.utils.io import read_jsonl


def validate_feature_metadata(metadata_path: str | Path) -> None:
    rows = read_jsonl(metadata_path)
    if not rows:
        raise ValueError(f"No feature metadata rows found: {metadata_path}")
    validate_feature_rows(rows)
