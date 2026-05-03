"""Time normalization for SEC availability and market labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

UTC = UTC
NY_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


def parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_et(value: str | datetime) -> datetime:
    return parse_datetime(value).astimezone(NY_TZ)


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass(frozen=True)
class TradingCalendar:
    """NYSE-like weekday calendar.

    This intentionally avoids inventing holiday data. If a price table is supplied,
    feature builders use actual traded dates from that table.
    """

    traded_dates: tuple[date, ...] | None = None

    def is_trading_day(self, value: date) -> bool:
        if self.traded_dates is not None:
            return value in set(self.traded_dates)
        return value.weekday() < 5

    def next_trading_day(self, value: date) -> date:
        current = value
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        return current

    def strictly_next_trading_day(self, value: date) -> date:
        return self.next_trading_day(value + timedelta(days=1))

    def previous_trading_day(self, value: date) -> date:
        current = value - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        return current

    def event_start_date(self, available_at: str | datetime) -> date:
        accepted_et = to_et(available_at)
        accepted_date = accepted_et.date()
        if self.is_trading_day(accepted_date) and accepted_et.time() < MARKET_CLOSE:
            return accepted_date
        return self.strictly_next_trading_day(accepted_date)

    def event_end_date(self, start: date, horizon: int) -> date:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        current = start
        for _ in range(horizon - 1):
            current = self.strictly_next_trading_day(current)
        return current
