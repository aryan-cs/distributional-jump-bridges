from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from cebt.data.factors import FactorRow
from cebt.data.prices import PriceBar
from cebt.features.build import _compound_market_return, build_feature_bundle, validate_feature_rows
from cebt.utils.time import TradingCalendar


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
            "controls_per_event": 3,
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
    assert len([row for row in bundle.rows if row["control_type"] == "same_ticker_no_event"]) == 3
    assert _control_windows_do_not_overlap(bundle.rows)
    assert all(row["feature_max_date"] < row["label_start_date"] for row in bundle.rows)
    assert np.all(np.isfinite(bundle.y))


def test_feature_bundle_builds_ff4_factor_residual_labels(tmp_path) -> None:
    bars = _bars("MSFT", date(2023, 1, 3), 130)
    market = _bars("SPY", date(2023, 1, 3), 130)
    factors = [
        FactorRow(
            date=bar.date,
            mkt_rf=0.001,
            smb=0.0002,
            hml=-0.0001,
            rf=0.00005,
            mom=0.0003,
            source_url="unit-test",
        )
        for bar in bars
    ]
    event = {
        "event_id": "event:ff4",
        "ticker": "MSFT",
        "cik": "0000789019",
        "company_name": "Microsoft Corp.",
        "accession_number": "0000789019-23-000001",
        "form_type": "8-K",
        "filing_date": "2023-03-28",
        "accepted_at": "2023-03-28T18:00:00+00:00",
        "available_at": "2023-03-28T18:00:00+00:00",
        "source_url": "https://www.sec.gov/",
    }
    config = {
        "seed": 11,
        "data": {
            "pre_window": 10,
            "horizon": 3,
            "control_blackout_days": 5,
            "controls_per_event": 1,
        },
        "labels": {
            "mode": "ff4",
            "factor_estimation_window": 20,
            "min_factor_observations": 8,
        },
        "features": {"embedding": {"dim": 8, "model_id": "test-hash"}},
        "training": {"split": {"train_until": "2023-12-31", "val_until": "2024-12-31"}},
    }

    bundle = build_feature_bundle(
        [event],
        {"MSFT": bars, "SPY": market},
        market,
        config,
        tmp_path,
        factor_rows=factors,
    )

    real_rows = [row for row in bundle.rows if row["control_type"] == "real_event"]
    assert real_rows
    assert real_rows[0]["label_mode"] == "ff4"
    assert real_rows[0]["label_source"] == "future_returns_and_factors_only"
    assert real_rows[0]["label_factor_model"] == "FF4"
    assert real_rows[0]["label_factor_observations"] >= 8


def test_market_forward_return_excludes_label_start_close_return() -> None:
    days = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5))
    calendar = TradingCalendar(days)
    market_returns = {
        days[0]: 0.50,
        days[1]: 0.10,
        days[2]: -0.05,
        days[3]: 0.02,
    }

    observed = _compound_market_return(days[0], days[3], market_returns, calendar)
    expected = (1.10 * 0.95 * 1.02) - 1.0

    assert observed == pytest.approx(expected)


def _control_windows_do_not_overlap(rows: list[dict]) -> bool:
    windows = [
        (
            date.fromisoformat(row["label_start_date"]),
            date.fromisoformat(row["label_end_date"]),
        )
        for row in rows
        if row["control_type"] == "same_ticker_no_event"
    ]
    for left_idx, (left_start, left_end) in enumerate(windows):
        for right_start, right_end in windows[left_idx + 1 :]:
            if left_start <= right_end and right_start <= left_end:
                return False
    return True


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
