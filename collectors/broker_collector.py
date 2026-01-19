"""
Broker Summary Collector - Collect broker activity data for all stocks
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock
)
from database.models import BrokerSummary
from config import DATASAHAM_API_KEY, API_RATE_LIMIT


class BrokerSummaryCollector:
    """Collect broker summary data for all stocks"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)

    def get_broker_summary(self, symbol: str) -> List[Dict]:
        """
        Get broker summary for a stock

        Args:
            symbol: Stock symbol

        Returns:
            List of broker activity records
        """
        try:
            result = self.api.emiten_broker_summary(symbol)
            # Handle various response structures
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                # Try common response patterns
                broker_list = result.get("broker_list", [])
                if not broker_list:
                    broker_list = result.get("data", {}).get("broker_list", [])
                if not broker_list:
                    broker_list = result.get("brokers", [])
                if not broker_list and "data" in result:
                    data = result.get("data")
                    if isinstance(data, list):
                        broker_list = data
                return broker_list
            return []
        except Exception as e:
            print(f"Error getting broker summary for {symbol}: {e}")
            return []

    def parse_broker_data(self, broker_data: List[Dict], date: datetime) -> List[Dict]:
        """Parse broker data into standardized format"""
        parsed = []
        for broker in broker_data:
            # Handle various field name patterns
            broker_code = (
                broker.get("broker_code") or
                broker.get("code") or
                broker.get("broker") or
                broker.get("brokerId")
            )
            if not broker_code:
                continue

            parsed.append({
                "date": date.date() if isinstance(date, datetime) else date,
                "broker_code": broker_code,
                "broker_name": broker.get("broker_name") or broker.get("name") or broker.get("brokerName"),
                "buy_value": broker.get("buy_value") or broker.get("bval") or broker.get("buyValue") or 0,
                "sell_value": broker.get("sell_value") or broker.get("sval") or broker.get("sellValue") or 0,
                "net_value": broker.get("net_value") or broker.get("nval") or broker.get("netValue") or 0,
                "buy_volume": broker.get("buy_volume") or broker.get("bvol") or broker.get("buyVolume") or 0,
                "sell_volume": broker.get("sell_volume") or broker.get("svol") or broker.get("sellVolume") or 0,
                "net_volume": broker.get("net_volume") or broker.get("nvol") or broker.get("netVolume") or 0,
                "buy_frequency": broker.get("buy_frequency") or broker.get("bfreq") or broker.get("buyFreq") or 0,
                "sell_frequency": broker.get("sell_frequency") or broker.get("sfreq") or broker.get("sellFreq") or 0,
            })
        return parsed

    def _bulk_upsert_broker_summaries(self, session, stock_id: int, broker_data: list):
        """Bulk insert or update broker summary records"""
        from sqlalchemy.dialects.sqlite import insert

        for data in broker_data:
            record = {
                "stock_id": stock_id,
                **data
            }
            stmt = insert(BrokerSummary).values(**record)
            update_fields = {k: stmt.excluded[k] for k in data.keys() if k not in ("date", "broker_code")}
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id", "date", "broker_code"],
                set_=update_fields
            )
            session.execute(stmt)

    def collect_and_save(
        self,
        symbols: List[str] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Collect and save broker summary data for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all active stocks.
            show_progress: Show progress bar

        Returns:
            Dict with collection statistics
        """
        stats = {"success": 0, "failed": 0, "records": 0}
        today = datetime.now()

        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(symbols, desc="Collecting broker summaries") if show_progress else symbols

        for symbol in iterator:
            try:
                broker_data = self.get_broker_summary(symbol)

                if broker_data:
                    parsed = self.parse_broker_data(broker_data, today)

                    if parsed:
                        with session_scope() as session:
                            stock = get_stock_by_symbol(session, symbol)
                            if stock:
                                self._bulk_upsert_broker_summaries(session, stock.id, parsed)
                                stats["records"] += len(parsed)
                                stats["success"] += 1
                            else:
                                stats["failed"] += 1
                    else:
                        stats["success"] += 1  # No broker data is valid
                else:
                    stats["success"] += 1  # No broker data is valid

                time.sleep(API_RATE_LIMIT)

            except Exception as e:
                print(f"Error collecting broker data for {symbol}: {e}")
                stats["failed"] += 1
                continue

        print(f"\nBroker collection complete: {stats['success']} stocks, {stats['records']} records")
        return stats

    def collect_today(self) -> Dict[str, int]:
        """Collect today's broker summary for all stocks"""
        return self.collect_and_save()


if __name__ == "__main__":
    from database import init_db
    init_db()

    collector = BrokerSummaryCollector()
    # Test with single stock
    result = collector.get_broker_summary("BBCA")
    print(f"BBCA broker data: {len(result) if result else 0} brokers")
    if result:
        print(f"Sample: {result[0] if result else 'N/A'}")
