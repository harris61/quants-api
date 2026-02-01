"""
Order Book Collector - Collect order book snapshots
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import session_scope, get_stock_by_symbol, Stock
from database.models import OrderBookSnapshot
from config import DATASAHAM_API_KEY, API_RATE_LIMIT


class OrderBookCollector:
    """Collect order book snapshot data for stocks"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)

    def _parse_datetime(self, value) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    def _extract_orderbook(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict) and "orderbook" in data:
                data = data.get("orderbook")
            return data if isinstance(data, dict) else {"data": data}
        if isinstance(result, list):
            return {"data": result}
        return {"data": result}

    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        try:
            return self.api.emiten_orderbook(symbol)
        except Exception as exc:
            print(f"Error getting orderbook for {symbol}: {exc}")
            return {}

    def _parse_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        bids = (
            payload.get("bids")
            or payload.get("bid")
            or payload.get("buy")
            or payload.get("bid_book")
            or payload.get("buy_book")
        )
        asks = (
            payload.get("asks")
            or payload.get("ask")
            or payload.get("sell")
            or payload.get("offer")
            or payload.get("sell_book")
        )
        source_ts = (
            payload.get("timestamp")
            or payload.get("datetime")
            or payload.get("time")
            or payload.get("date")
            or payload.get("last_update")
        )

        return {
            "bids": bids,
            "asks": asks,
            "source_timestamp": self._parse_datetime(source_ts),
        }

    def collect_and_save(
        self,
        symbols: List[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        stats = {"success": 0, "failed": 0, "records": 0}

        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(symbols, desc="Collecting orderbook") if show_progress else symbols

        for symbol in iterator:
            try:
                result = self.get_orderbook(symbol)
                payload = self._extract_orderbook(result)
                parsed = self._parse_snapshot(payload)

                # Use WIB for captured_at/date to align with IDX trading date
                captured_at = datetime.utcnow() + timedelta(hours=7)
                with session_scope() as session:
                    stock = get_stock_by_symbol(session, symbol)
                    if not stock:
                        stats["failed"] += 1
                        continue

                    snapshot = OrderBookSnapshot(
                        stock_id=stock.id,
                        captured_at=captured_at,
                        date=captured_at.date(),
                        source_timestamp=parsed.get("source_timestamp"),
                        bids_json=json.dumps(parsed.get("bids"), ensure_ascii=True),
                        asks_json=json.dumps(parsed.get("asks"), ensure_ascii=True),
                        raw_json=json.dumps(payload, ensure_ascii=True),
                    )
                    session.add(snapshot)
                    stats["records"] += 1
                    stats["success"] += 1

                self.api.smart_sleep()
            except Exception as exc:
                print(f"Error collecting orderbook for {symbol}: {exc}")
                stats["failed"] += 1
                continue

        print(
            f"\nOrderbook collection complete: "
            f"{stats['success']} stocks, {stats['records']} records"
        )
        return stats


if __name__ == "__main__":
    from database import init_db
    init_db()

    collector = OrderBookCollector()
    data = collector.get_orderbook("BBCA")
    print("Sample orderbook payload keys:", list(data.keys()) if isinstance(data, dict) else type(data))
