"""
Basic trading day utilities (weekend-only).
"""

from datetime import date, datetime, timedelta

from config import IDX_HOLIDAYS

_HOLIDAY_SET = None


def _get_holiday_set() -> set:
    """Return cached holiday dates."""
    global _HOLIDAY_SET
    if _HOLIDAY_SET is not None:
        return _HOLIDAY_SET

    holidays = set()
    for item in IDX_HOLIDAYS:
        if isinstance(item, date):
            holidays.add(item)
            continue
        if isinstance(item, str):
            try:
                holidays.add(datetime.strptime(item, "%Y-%m-%d").date())
            except ValueError:
                continue

    _HOLIDAY_SET = holidays
    return _HOLIDAY_SET


def is_trading_day(check_date: date) -> bool:
    """Return True for IDX trading days."""
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    if check_date.weekday() >= 5:
        return False
    if check_date in _get_holiday_set():
        return False
    return True


def next_trading_day(from_date: date) -> date:
    """Return the next trading day after from_date."""
    if isinstance(from_date, datetime):
        from_date = from_date.date()
    next_day = from_date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day
