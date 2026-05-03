from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from cebt.data.prices import PriceBar
from cebt.features.build import build_feature_bundle, validate_feature_rows


def test_feature_builder_rejects_leaky_rows() -> None:
    with pytest.raises(ValueError, match="Feature leakage"):
        validate_feature_rows(
            [
                {
                    "sample_id": "bad",
                    "feature_max_date": "2025-01-03",
                    "label_start_date": "2025-01-03",
                    "label_source": "future_returns_only",
                }
            ]
        )


def test_feature_bundle_builds_real_events_and_controls(tmp_path) -> None:
    bars = _bars("AAPL", date(2024, 1, 2), 110)
    market = _bars("SPY", date(2024, 1, 2), 110)
    event = {
        "event_id": "event:test",
        "ticker": "AAPL",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "accession_number": "0000320193-24-000001",
        "form_type": "8-K",
        "filing_date": "2024-03-20",
        "accepted_at": "2024-03-20T21:01:00+00:00",
        "available_at": "2024-03-20T21:01:00+00:00",
        "source_url": "https://www.sec.gov/",
    }
    config = {
        "seed": 7,
        "data": {
            "pre_window": 5,
            "horizon": 3,
            "control_blackout_days": 5,
            "controls_per_event": 1,
        },
        "features": {"embedding": {"dim": 16, "model_id": "test-hash"}},
        "training": {"split": {"train_until": "2024-12-31", "val_until": "2025-12-31"}},
    }
    bundle = build_feature_bundle([event], {"AAPL": bars, "SPY": market}, market, config, tmp_path)
    assert bundle.x_pre.shape[1:] == (5, 8)
    assert bundle.event_embedding.shape[1] == 16
    assert bundle.metadata.shape[1] == 6
    assert bundle.y.shape[1] == 3
    assert {row["control_type"] for row in bundle.rows} == {"real_event", "same_ticker_no_event"}
    assert all(row["feature_max_date"] < row["label_start_date"] for row in bundle.rows)
    assert np.all(np.isfinite(bundle.y))


def _bars(ticker: str, start: date, count: int) -> list[PriceBar]:
    rows = []
    current = start
    observed = 0
    while observed < count:
        if current.weekday() < 5:
            base = 100.0 + observed * 0.2
            rows.append(
                PriceBar(
                    ticker=ticker,
                    date=current,
                    open=base,
                    high=base + 1.0,
                    low=base - 1.0,
                    close=base + 0.3,
                    volume=1_000_000 + observed * 1000,
                    source_url="unit-test",
                )
            )
            observed += 1
        current += timedelta(days=1)
    return rows
