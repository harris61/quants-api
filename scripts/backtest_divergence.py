"""
Backtest divergence picks for next-trading-day open-to-close returns.
"""

import argparse
import sqlite3
from pathlib import Path


def parse_codes(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.split(",")]
    return [p for p in parts if p]


def fmt_float(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "NULL"
    return f"{value:.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "NULL"
    return f"{value * 100:.{decimals}f}%"


def _get_trading_dates(cur: sqlite3.Cursor, start: str, end: str) -> list[str]:
    from datetime import datetime
    from utils.holidays import is_trading_day

    cur.execute(
        """
        SELECT DISTINCT date
        FROM daily_prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """,
        (start, end),
    )
    dates = []
    for (date_str,) in cur.fetchall():
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if is_trading_day(dt):
            dates.append(date_str)
    return dates


def _get_price_change(
    cur: sqlite3.Cursor,
    symbol: str,
    date_str: str,
    lookback_days: int,
) -> float | None:
    """Return price drop % from the recent high within the lookback window."""
    cur.execute(
        """
        SELECT dp.close
        FROM daily_prices dp
        JOIN stocks s ON s.id = dp.stock_id
        WHERE s.symbol = ? AND dp.date <= ?
        ORDER BY dp.date DESC
        LIMIT ?
        """,
        (symbol, date_str, lookback_days + 1),
    )
    rows = [r[0] for r in cur.fetchall() if r[0] is not None]
    if len(rows) < 2:
        return None
    current = rows[0]
    recent_high = max(rows)
    if recent_high == 0:
        return None
    return (current - recent_high) / recent_high


def _get_top_divergence(
    cur: sqlite3.Cursor,
    date_str: str,
    smart_codes: list[str],
    retail_codes: list[str],
    limit: int,
    lookback_days: int,
    max_drop: float | None = None,
) -> list[tuple[str, float | None]]:
    if lookback_days < 1:
        lookback_days = 1

    if lookback_days == 1:
        smart_date_clause = "bs.date = ?"
        retail_date_clause = "bs.date = ?"
        smart_date_params = [date_str]
        retail_date_params = [date_str]
    else:
        smart_date_clause = "bs.date BETWEEN date(?, ?) AND ?"
        retail_date_clause = "bs.date BETWEEN date(?, ?) AND ?"
        offset = f"-{lookback_days - 1} day"
        smart_date_params = [date_str, offset, date_str]
        retail_date_params = [date_str, offset, date_str]

    smart_placeholders = ",".join(["?"] * len(smart_codes))
    retail_placeholders = ",".join(["?"] * len(retail_codes))

    query = f"""
    WITH smart AS (
      SELECT bs.stock_id,
             SUM(bs.buy_value) AS smart_buy,
             SUM(bs.sell_value) AS smart_sell
      FROM broker_summaries bs
      WHERE {smart_date_clause}
        AND bs.broker_code IN ({smart_placeholders})
      GROUP BY bs.stock_id
    ),
    retail AS (
      SELECT bs.stock_id,
             SUM(bs.buy_value) AS retail_buy,
             SUM(bs.sell_value) AS retail_sell
      FROM broker_summaries bs
      WHERE {retail_date_clause}
        AND bs.broker_code IN ({retail_placeholders})
      GROUP BY bs.stock_id
    )
    SELECT s.symbol,
           CASE WHEN s.market_cap IS NOT NULL AND s.market_cap != 0
                THEN ((COALESCE(sm.smart_buy,0) - COALESCE(sm.smart_sell,0))
                      - (COALESCE(rt.retail_buy,0) - COALESCE(rt.retail_sell,0))) / s.market_cap
                ELSE NULL END AS net_divergence
    FROM stocks s
    LEFT JOIN smart sm ON sm.stock_id = s.id
    LEFT JOIN retail rt ON rt.stock_id = s.id
    WHERE s.is_active = 1
      AND (sm.smart_buy IS NOT NULL OR sm.smart_sell IS NOT NULL
           OR rt.retail_buy IS NOT NULL OR rt.retail_sell IS NOT NULL)
    ORDER BY net_divergence DESC
    LIMIT ?
    """

    sql_limit = limit * 3 if max_drop is not None else limit
    params = [
        *smart_date_params,
        *smart_codes,
        *retail_date_params,
        *retail_codes,
        sql_limit,
    ]
    cur.execute(query, params)
    results = [(row[0], row[1]) for row in cur.fetchall()]

    if max_drop is not None:
        filtered = []
        for symbol, net_div in results:
            change = _get_price_change(cur, symbol, date_str, lookback_days)
            if change is not None and change < -max_drop:
                continue
            filtered.append((symbol, net_div))
        results = filtered[:limit]

    return results


def _get_close(
    cur: sqlite3.Cursor,
    symbol: str,
    date_str: str,
) -> float | None:
    cur.execute(
        """
        SELECT dp.close
        FROM daily_prices dp
        JOIN stocks s ON s.id = dp.stock_id
        WHERE s.symbol = ? AND dp.date = ?
        """,
        (symbol, date_str),
    )
    row = cur.fetchone()
    if not row:
        return None
    return row[0]


def _get_next_close(
    cur: sqlite3.Cursor,
    symbol: str,
    date_str: str,
) -> tuple[str | None, float | None]:
    cur.execute(
        """
        SELECT dp.date, dp.close
        FROM daily_prices dp
        JOIN stocks s ON s.id = dp.stock_id
        WHERE s.symbol = ? AND dp.date > ?
        ORDER BY dp.date
        LIMIT 1
        """,
        (symbol, date_str),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def run(
    start: str,
    end: str,
    smart_codes: list[str],
    retail_codes: list[str],
    top: int,
    lookback_days: int,
    max_drop: float | None = None,
) -> None:
    if not smart_codes or not retail_codes:
        raise SystemExit("Both --smart and --retail broker code lists are required.")

    db_path = Path(__file__).resolve().parents[1] / "database" / "quants.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    dates = _get_trading_dates(cur, start, end)
    if not dates:
        print("No trading dates found in range.")
        conn.close()
        return

    overall_returns = []
    print("date | next_date | symbol | net_divergence | prev_close | close | return")
    for date_str in dates:
        picks = _get_top_divergence(cur, date_str, smart_codes, retail_codes, top, lookback_days, max_drop)
        if not picks:
            continue

        day_returns = []
        for symbol, net_div in picks:
            prev_close = _get_close(cur, symbol, date_str)
            next_date, close_px = _get_next_close(cur, symbol, date_str)
            if prev_close is None or close_px is None or prev_close == 0:
                ret = None
            else:
                ret = (close_px - prev_close) / prev_close
                day_returns.append(ret)
                overall_returns.append(ret)

            print(
                f"{date_str} | {next_date} | {symbol} | {fmt_float(net_div, 8)} | "
                f"{fmt_float(prev_close)} | {fmt_float(close_px)} | {fmt_pct(ret)}"
            )

        if day_returns:
            avg_ret = sum(day_returns) / len(day_returns)
            print(f"{date_str} | {next_date} | AVG | - | - | - | {fmt_pct(avg_ret)}")

    if overall_returns:
        overall_avg = sum(overall_returns) / len(overall_returns)
        print(f"OVERALL | - | AVG | - | - | - | {fmt_pct(overall_avg)}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest divergence picks (next-trading-day open-to-close returns)."
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--smart", required=True, help="Comma-separated smart broker codes")
    parser.add_argument("--retail", required=True, help="Comma-separated retail broker codes")
    parser.add_argument("--top", type=int, default=5, help="Top N picks per day")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="Accumulate divergence over the previous N calendar days (default 1)",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=None,
        help="Filter out stocks that dropped more than this %% over the lookback period (e.g. 10 for 10%%)",
    )
    args = parser.parse_args()

    run(
        args.start,
        args.end,
        parse_codes(args.smart),
        parse_codes(args.retail),
        args.top,
        args.lookback_days,
        max_drop=args.max_drop / 100 if args.max_drop is not None else None,
    )


if __name__ == "__main__":
    main()
