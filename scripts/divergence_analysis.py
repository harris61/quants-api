"""
Divergence analysis based on broker summaries and market cap.
"""

import argparse
import sqlite3
from pathlib import Path


def parse_codes(raw: str) -> list[str]:
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.split(",")]
    return [p for p in parts if p]


def fmt_int(value: float | None) -> str:
    if value is None:
        return "NULL"
    return f"{int(round(value)):,}"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NULL"
    return f"{value:.8f}"


def run(date_str: str, smart_codes: list[str], retail_codes: list[str], limit: int) -> None:
    db_path = Path(__file__).resolve().parents[1] / "database" / "quants.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not smart_codes or not retail_codes:
        raise SystemExit("Both --smart and --retail broker code lists are required.")

    smart_placeholders = ",".join(["?"] * len(smart_codes))
    retail_placeholders = ",".join(["?"] * len(retail_codes))

    query = f"""
    WITH smart AS (
      SELECT bs.stock_id,
             SUM(bs.buy_value) AS smart_buy,
             SUM(bs.sell_value) AS smart_sell
      FROM broker_summaries bs
      WHERE bs.date = ?
        AND bs.broker_code IN ({smart_placeholders})
      GROUP BY bs.stock_id
    ),
    retail AS (
      SELECT bs.stock_id,
             SUM(bs.buy_value) AS retail_buy,
             SUM(bs.sell_value) AS retail_sell
      FROM broker_summaries bs
      WHERE bs.date = ?
        AND bs.broker_code IN ({retail_placeholders})
      GROUP BY bs.stock_id
    )
    SELECT s.symbol,
           ((COALESCE(sm.smart_buy,0) - COALESCE(sm.smart_sell,0))
             - (COALESCE(rt.retail_buy,0) - COALESCE(rt.retail_sell,0))) AS raw_net_divergence,
           s.market_cap,
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

    params = [date_str, *smart_codes, date_str, *retail_codes, limit]
    cur.execute(query, params)
    rows = cur.fetchall()

    print(f"Top {limit} net divergence for {date_str}")
    print("symbol | raw_net_divergence | market_cap | net_divergence")
    for symbol, raw_net, market_cap, net_div in rows:
        print(f"{symbol} | {fmt_int(raw_net)} | {fmt_int(market_cap)} | {fmt_float(net_div)}")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Broker divergence analysis.")
    parser.add_argument("--date", required=True, help="Date for analysis (YYYY-MM-DD)")
    parser.add_argument("--smart", required=True, help="Comma-separated smart broker codes")
    parser.add_argument("--retail", required=True, help="Comma-separated retail broker codes")
    parser.add_argument("--limit", type=int, default=10, help="Top N results")
    args = parser.parse_args()

    run(args.date, parse_codes(args.smart), parse_codes(args.retail), args.limit)


if __name__ == "__main__":
    main()
