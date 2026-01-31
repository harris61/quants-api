"""
Divergence analysis based on broker summaries and market cap.
"""

import argparse
import sqlite3
import json
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


def _parse_number(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("+", "").strip()
        if cleaned == "":
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def _sum_levels(levels) -> float | None:
    if levels is None:
        return None
    if isinstance(levels, dict):
        levels = list(levels.values())
    if not isinstance(levels, list):
        return None
    total = 0.0
    for level in levels:
        if isinstance(level, dict):
            qty = (
                level.get("qty")
                or level.get("quantity")
                or level.get("lot")
                or level.get("lots")
                or level.get("volume")
                or level.get("size")
                or level.get("amount")
                or level.get("shares")
                or level.get("value")
            )
            total += _parse_number(qty)
        elif isinstance(level, (list, tuple)):
            if len(level) >= 2:
                total += _parse_number(level[1])
            elif len(level) == 1:
                total += _parse_number(level[0])
        else:
            total += _parse_number(level)
    return total


def _load_orderbook_totals(cur: sqlite3.Cursor, date_str: str) -> dict[int, tuple[float | None, float | None]]:
    query = """
    SELECT ob.stock_id, ob.bids_json, ob.asks_json
    FROM orderbook_snapshots ob
    WHERE ob.date = ?
      AND ob.captured_at = (
        SELECT MAX(ob2.captured_at)
        FROM orderbook_snapshots ob2
        WHERE ob2.stock_id = ob.stock_id AND ob2.date = ?
      )
    """
    cur.execute(query, (date_str, date_str))
    rows = cur.fetchall()
    totals = {}
    for stock_id, bids_json, asks_json in rows:
        try:
            bids = json.loads(bids_json) if bids_json else None
        except json.JSONDecodeError:
            bids = None
        try:
            asks = json.loads(asks_json) if asks_json else None
        except json.JSONDecodeError:
            asks = None
        totals[stock_id] = (_sum_levels(bids), _sum_levels(asks))
    return totals


def run(
    date_str: str,
    smart_codes: list[str],
    retail_codes: list[str],
    limit: int,
    orderbook_bid_gt_ask: bool = False,
    show_orderbook_totals: bool = False,
) -> None:
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

    orderbook_totals = {}
    if orderbook_bid_gt_ask:
        orderbook_totals = _load_orderbook_totals(cur, date_str)
        filtered = []
        for symbol, raw_net, market_cap, net_div in rows:
            cur.execute("SELECT id FROM stocks WHERE symbol = ?", (symbol,))
            stock_row = cur.fetchone()
            if not stock_row:
                continue
            stock_id = stock_row[0]
            totals = orderbook_totals.get(stock_id)
            if not totals:
                continue
            bid_total, ask_total = totals
            if bid_total is None or ask_total is None:
                continue
            if bid_total > ask_total:
                filtered.append((symbol, raw_net, market_cap, net_div))
        rows = filtered

    print(f"Top {limit} net divergence for {date_str}")
    if show_orderbook_totals and orderbook_bid_gt_ask:
        print("symbol | raw_net_divergence | market_cap | net_divergence | bid_total | ask_total")
        for symbol, raw_net, market_cap, net_div in rows:
            cur.execute("SELECT id FROM stocks WHERE symbol = ?", (symbol,))
            stock_row = cur.fetchone()
            bid_total, ask_total = (None, None)
            if stock_row:
                totals = orderbook_totals.get(stock_row[0])
                if totals:
                    bid_total, ask_total = totals
            print(
                f"{symbol} | {fmt_int(raw_net)} | {fmt_int(market_cap)} | {fmt_float(net_div)} | "
                f"{fmt_int(bid_total)} | {fmt_int(ask_total)}"
            )
    else:
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
    parser.add_argument(
        "--orderbook-bid-gt-ask",
        action="store_true",
        help="Filter to stocks with total bids greater than total asks (latest snapshot on date)",
    )
    parser.add_argument(
        "--orderbook-totals",
        action="store_true",
        help="Show bid/ask totals when using orderbook filter",
    )
    args = parser.parse_args()

    run(
        args.date,
        parse_codes(args.smart),
        parse_codes(args.retail),
        args.limit,
        orderbook_bid_gt_ask=args.orderbook_bid_gt_ask,
        show_orderbook_totals=args.orderbook_totals,
    )


if __name__ == "__main__":
    main()
