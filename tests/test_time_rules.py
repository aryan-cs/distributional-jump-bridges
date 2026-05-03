from __future__ import annotations

from cebt.utils.time import TradingCalendar


def test_after_hours_event_maps_to_next_trading_day() -> None:
    calendar = TradingCalendar()
    assert calendar.event_start_date("2025-05-01T21:00:00+00:00").isoformat() == "2025-05-02"


def test_before_close_event_maps_to_same_trading_day() -> None:
    calendar = TradingCalendar()
    assert calendar.event_start_date("2025-05-01T18:00:00+00:00").isoformat() == "2025-05-01"
