"""Leakage-safe tensor feature construction for CEBT."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Any

import numpy as np

from cebt.data.prices import PriceBar, close_returns
from cebt.features.embeddings import HashingEmbedder, embed_texts_with_cache, zero_embedding
from cebt.utils.hashing import stable_id
from cebt.utils.io import write_json, write_jsonl
from cebt.utils.time import TradingCalendar, parse_date

PRICE_FEATURES = 8
METADATA_FEATURES = 6


@dataclass(frozen=True)
class FeatureBundle:
    x_pre: np.ndarray
    event_embedding: np.ndarray
    metadata: np.ndarray
    y: np.ndarray
    is_event: np.ndarray
    split: np.ndarray
    event_ids: list[str]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class PriceContext:
    bars: list[PriceBar]
    calendar: TradingCalendar
    bars_by_date: dict[date, PriceBar]
    ticker_returns: dict[date, float]
    volume_by_date: dict[date, float]
    ordered_dates: list[date]
    date_index: dict[date, int]
    market_returns: dict[date, float]


def build_feature_bundle(
    events: list[dict[str, Any]],
    prices_by_ticker: dict[str, list[PriceBar]],
    market_bars: list[PriceBar],
    config: dict[str, Any],
    output_dir: str | Path,
) -> FeatureBundle:
    data_config = config.get("data", {})
    feature_config = config.get("features", {})
    embedding_config = feature_config.get("embedding", {})
    pre_window = int(data_config.get("pre_window", 40))
    horizon = int(data_config.get("horizon", 5))
    blackout = int(data_config.get("control_blackout_days", 10))
    controls_per_event = int(data_config.get("controls_per_event", 1))
    embedding_dim = int(embedding_config.get("dim", 256))
    embedder = HashingEmbedder(
        dim=embedding_dim, model_id=embedding_config.get("model_id", "cebt.hashing-v1")
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    text_items = []
    for event in events:
        text = _read_event_text(event)
        text_items.append((event["event_id"], text))
    embeddings = embed_texts_with_cache(text_items, embedder, output_path / "embedding_cache.jsonl")

    market_returns = close_returns(market_bars)
    rows: list[dict[str, Any]] = []
    x_rows: list[np.ndarray] = []
    e_rows: list[np.ndarray] = []
    m_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    is_event_rows: list[float] = []
    event_ids: list[str] = []

    contexts = {
        ticker: _price_context(bars, market_returns)
        for ticker, bars in prices_by_ticker.items()
        if bars
    }

    real_event_starts: dict[str, set[date]] = {}
    for event in events:
        ticker = event["ticker"]
        context = contexts.get(ticker)
        if context is None:
            continue
        start = context.calendar.event_start_date(event["available_at"])
        real_event_starts.setdefault(ticker, set()).add(start)

    control_candidates = {
        ticker: _control_candidates(
            contexts[ticker],
            starts,
            pre_window,
            horizon,
            blackout,
        )
        for ticker, starts in real_event_starts.items()
        if ticker in contexts
    }

    for event in events:
        ticker = event["ticker"]
        context = contexts.get(ticker)
        if context is None:
            continue
        sample = _build_sample(
            event=event,
            sample_id=event["event_id"],
            control_type="real_event",
            event_embedding=embeddings[event["event_id"]],
            context=context,
            pre_window=pre_window,
            horizon=horizon,
            forced_start=None,
        )
        if sample:
            _append_sample(sample, rows, x_rows, e_rows, m_rows, y_rows, is_event_rows, event_ids)
        for control_start in _control_dates(
            event,
            control_candidates.get(ticker, []),
            controls_per_event,
        ):
            control_id = stable_id(event["event_id"], control_start.isoformat(), prefix="control")
            control_sample = _build_sample(
                event=event,
                sample_id=control_id,
                control_type="same_ticker_no_event",
                event_embedding=zero_embedding(embedding_dim),
                context=context,
                pre_window=pre_window,
                horizon=horizon,
                forced_start=control_start,
            )
            if control_sample:
                _append_sample(
                    control_sample, rows, x_rows, e_rows, m_rows, y_rows, is_event_rows, event_ids
                )

    if not rows:
        raise ValueError("No feature rows were built; check event dates, prices, and pre_window.")

    split = np.asarray([_split_for_row(row, config) for row in rows], dtype=np.int64)
    bundle = FeatureBundle(
        x_pre=np.stack(x_rows).astype(np.float32),
        event_embedding=np.stack(e_rows).astype(np.float32),
        metadata=np.stack(m_rows).astype(np.float32),
        y=np.stack(y_rows).astype(np.float32),
        is_event=np.asarray(is_event_rows, dtype=np.float32),
        split=split,
        event_ids=event_ids,
        rows=rows,
    )
    validate_feature_rows(bundle.rows)
    np.savez_compressed(
        output_path / "features.npz",
        x_pre=bundle.x_pre,
        event_embedding=bundle.event_embedding,
        metadata=bundle.metadata,
        y=bundle.y,
        is_event=bundle.is_event,
        split=bundle.split,
        event_ids=np.asarray(bundle.event_ids, dtype=object),
    )
    write_jsonl(output_path / "features_metadata.jsonl", bundle.rows)
    write_json(
        output_path / "features_summary.json",
        {
            "rows": len(rows),
            "real_events": int(np.sum(bundle.is_event)),
            "controls": int(len(rows) - np.sum(bundle.is_event)),
            "pre_window": pre_window,
            "horizon": horizon,
            "price_features": PRICE_FEATURES,
            "metadata_features": METADATA_FEATURES,
        },
    )
    return bundle


def load_feature_bundle(feature_path: str | Path, metadata_path: str | Path) -> FeatureBundle:
    arrays = np.load(feature_path, allow_pickle=True)
    rows = []
    if Path(metadata_path).exists():
        import json

        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    return FeatureBundle(
        x_pre=arrays["x_pre"],
        event_embedding=arrays["event_embedding"],
        metadata=arrays["metadata"],
        y=arrays["y"],
        is_event=arrays["is_event"],
        split=arrays["split"],
        event_ids=[str(item) for item in arrays["event_ids"].tolist()],
        rows=rows,
    )


def validate_feature_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        feature_max = parse_date(row["feature_max_date"])
        label_start = parse_date(row["label_start_date"])
        if feature_max >= label_start:
            raise ValueError(
                f"Feature leakage for {row['sample_id']}: {feature_max} >= {label_start}"
            )
        if row.get("label_source") != "future_returns_only":
            raise ValueError(f"Invalid label source for {row['sample_id']}")


def _price_context(bars: list[PriceBar], market_returns: dict[date, float]) -> PriceContext:
    ordered_bars = sorted(bars, key=lambda row: row.date)
    ordered_dates = [bar.date for bar in ordered_bars]
    return PriceContext(
        bars=ordered_bars,
        calendar=TradingCalendar(tuple(ordered_dates)),
        bars_by_date={bar.date: bar for bar in ordered_bars},
        ticker_returns=close_returns(ordered_bars),
        volume_by_date={bar.date: bar.volume for bar in ordered_bars},
        ordered_dates=ordered_dates,
        date_index={day: idx for idx, day in enumerate(ordered_dates)},
        market_returns=market_returns,
    )


def _build_sample(
    event: dict[str, Any],
    sample_id: str,
    control_type: str,
    event_embedding: np.ndarray,
    context: PriceContext,
    pre_window: int,
    horizon: int,
    forced_start: date | None,
) -> dict[str, Any] | None:
    start = forced_start or context.calendar.event_start_date(event["available_at"])
    end = context.calendar.event_end_date(start, horizon)
    if start not in context.bars_by_date or end not in context.bars_by_date:
        return None
    pre_dates = _pre_dates(context.calendar, start, pre_window)
    if any(day not in context.bars_by_date for day in pre_dates):
        return None
    x_pre = _price_window_features(pre_dates, context)
    if x_pre is None:
        return None
    y = _label_vector(start, end, context)
    if y is None:
        return None
    feature_max = pre_dates[-1]
    row = {
        "sample_id": sample_id,
        "event_id": event["event_id"],
        "ticker": event["ticker"],
        "cik": event["cik"],
        "accession_number": event["accession_number"],
        "form_type": event["form_type"] if control_type == "real_event" else "CONTROL",
        "control_type": control_type,
        "available_at": event["available_at"],
        "label_start_date": start.isoformat(),
        "label_end_date": end.isoformat(),
        "feature_min_date": pre_dates[0].isoformat(),
        "feature_max_date": feature_max.isoformat(),
        "label_source": "future_returns_only",
        "x_pre": x_pre,
        "event_embedding": event_embedding,
        "metadata": _metadata(event, control_type, start),
        "y": y,
        "is_event": 1.0 if control_type == "real_event" else 0.0,
    }
    return row


def _append_sample(
    sample: dict[str, Any],
    rows: list[dict[str, Any]],
    x_rows: list[np.ndarray],
    e_rows: list[np.ndarray],
    m_rows: list[np.ndarray],
    y_rows: list[np.ndarray],
    is_event_rows: list[float],
    event_ids: list[str],
) -> None:
    x_rows.append(sample.pop("x_pre"))
    e_rows.append(sample.pop("event_embedding"))
    m_rows.append(sample.pop("metadata"))
    y_rows.append(sample.pop("y"))
    is_event_rows.append(float(sample["is_event"]))
    event_ids.append(sample["sample_id"])
    rows.append(sample)


def _read_event_text(event: dict[str, Any]) -> str:
    text_path = event.get("text_path")
    if text_path and Path(text_path).exists():
        return Path(text_path).read_text(encoding="utf-8", errors="replace")
    return " ".join(
        str(event.get(key, ""))
        for key in ("company_name", "form_type", "filing_date", "accession_number")
    )


def _pre_dates(calendar: TradingCalendar, start: date, count: int) -> list[date]:
    dates = []
    current = start
    for _ in range(count):
        current = calendar.previous_trading_day(current)
        dates.append(current)
    return list(reversed(dates))


def _price_window_features(
    pre_dates: list[date],
    context: PriceContext,
) -> np.ndarray | None:
    values = []
    for day in pre_dates:
        if day not in context.ticker_returns or day not in context.bars_by_date:
            return None
        trailing_5 = _trailing_values(context, context.ticker_returns, day, 5)
        trailing_20 = _trailing_values(context, context.ticker_returns, day, 20)
        trailing_volumes = _trailing_values(context, context.volume_by_date, day, 20)
        if not trailing_5 or len(trailing_20) < 2 or not trailing_volumes:
            return None
        ret_1 = context.ticker_returns[day]
        ret_5 = float(np.prod([1.0 + value for value in trailing_5]) - 1.0)
        ret_20 = float(np.prod([1.0 + value for value in trailing_20]) - 1.0)
        realized_vol_20 = float(pstdev(trailing_20))
        volume = context.volume_by_date[day]
        volume_mean = float(np.mean(trailing_volumes))
        volume_std = float(np.std(trailing_volumes))
        volume_z = 0.0 if volume_std == 0 else float((volume - volume_mean) / volume_std)
        market_ret = context.market_returns.get(day, 0.0)
        rel_ret = ret_1 - market_ret
        close_to_open = context.bars_by_date[day].close / context.bars_by_date[day].open - 1.0
        values.append(
            [ret_1, ret_5, ret_20, realized_vol_20, volume_z, market_ret, rel_ret, close_to_open]
        )
    return np.asarray(values, dtype=np.float32)


def _label_vector(
    start: date,
    end: date,
    context: PriceContext,
) -> np.ndarray | None:
    if (
        start not in context.bars_by_date
        or end not in context.bars_by_date
        or context.bars_by_date[start].close <= 0
    ):
        return None
    forward_return = context.bars_by_date[end].close / context.bars_by_date[start].close - 1.0
    market_forward = _compound_market_return(
        start,
        end,
        context.market_returns,
        context.calendar,
    )
    abnormal = forward_return - market_forward
    pre_returns = _trailing_values(
        context,
        context.ticker_returns,
        start,
        20,
        include_current=False,
    )
    post_returns = _window_values(context.ticker_returns, start, end, context.calendar)
    pre_volume = _trailing_values(
        context,
        context.volume_by_date,
        start,
        20,
        include_current=False,
    )
    post_volume = _window_values(context.volume_by_date, start, end, context.calendar)
    if len(pre_returns) < 2 or len(post_returns) < 2 or not pre_volume or not post_volume:
        return None
    volatility_jump = float(pstdev(post_returns) - pstdev(pre_returns))
    pre_volume_mean = float(np.mean(pre_volume))
    volume_jump = (
        0.0 if pre_volume_mean == 0 else float(np.mean(post_volume) / pre_volume_mean - 1.0)
    )
    return np.asarray([abnormal, volatility_jump, volume_jump], dtype=np.float32)


def _metadata(event: dict[str, Any], control_type: str, start: date) -> np.ndarray:
    form = event.get("form_type", "")
    days_since_2010 = (start - date(2010, 1, 1)).days / 3650.0
    return np.asarray(
        [
            1.0 if control_type == "real_event" else 0.0,
            1.0 if control_type != "real_event" else 0.0,
            1.0 if form == "8-K" and control_type == "real_event" else 0.0,
            1.0 if form == "10-Q" and control_type == "real_event" else 0.0,
            1.0 if form == "10-K" and control_type == "real_event" else 0.0,
            days_since_2010,
        ],
        dtype=np.float32,
    )


def _control_candidates(
    context: PriceContext,
    real_event_starts: set[date],
    pre_window: int,
    horizon: int,
    blackout: int,
) -> list[date]:
    blocked = {
        event_day + timedelta(days=offset)
        for event_day in real_event_starts
        for offset in range(-blackout, blackout + 1)
    }
    candidates = []
    traded = context.ordered_dates
    for day in traded[pre_window + 25 : -horizon - 2]:
        if day in blocked:
            continue
        try:
            end = context.calendar.event_end_date(day, horizon)
        except ValueError:
            continue
        if end in context.bars_by_date:
            candidates.append(day)
    return candidates


def _control_dates(
    event: dict[str, Any],
    candidates: list[date],
    count: int,
) -> list[date]:
    if not candidates:
        return []
    offset = int(stable_id(event["event_id"], prefix="offset").split(":")[1], 16) % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    return ordered[:count]


def _trailing_values(
    context: PriceContext,
    values_by_date: dict[date, float],
    day: date,
    count: int,
    include_current: bool = True,
) -> list[float]:
    idx = context.date_index.get(day)
    if idx is None:
        return []
    end = idx + 1 if include_current else idx
    start = max(0, end - count)
    return [
        values_by_date[value]
        for value in context.ordered_dates[start:end]
        if value in values_by_date
    ]


def _window_values(
    values_by_date: dict[date, float],
    start: date,
    end: date,
    calendar: TradingCalendar,
) -> list[float]:
    values = []
    current = start
    while current <= end:
        if current in values_by_date:
            values.append(values_by_date[current])
        current = calendar.strictly_next_trading_day(current)
    return values


def _compound_market_return(
    start: date,
    end: date,
    market_returns: dict[date, float],
    calendar: TradingCalendar,
) -> float:
    total = 1.0
    observed = 0
    current = start
    while current <= end:
        if current in market_returns:
            total *= 1.0 + market_returns[current]
            observed += 1
        current = calendar.strictly_next_trading_day(current)
    return 0.0 if observed == 0 else total - 1.0


def _split_for_row(row: dict[str, Any], config: dict[str, Any]) -> int:
    split = config.get("training", {}).get("split", {})
    train_until = parse_date(split.get("train_until", "2023-12-31"))
    val_until = parse_date(split.get("val_until", "2024-12-31"))
    label_start = parse_date(row["label_start_date"])
    if label_start <= train_until:
        return 0
    if label_start <= val_until:
        return 1
    return 2
