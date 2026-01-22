"""
Basic trading day utilities (weekend-only).
"""

from datetime import date, timedelta


def is_trading_day(check_date: date) -> bool:
    """Return True for weekdays (Mon-Fri)."""
    return check_date.weekday() < 5


def next_trading_day(from_date: date) -> date:
    """Return the next weekday after from_date."""
    next_day = from_date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day
