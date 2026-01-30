"""
Broker Summary Collector - Collect broker activity data for all stocks
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock
)
from database.models import BrokerSummary
from config import DATASAHAM_API_KEY, API_RATE_LIMIT
from utils.brokers import load_broker_categories


class BrokerSummaryCollector:
    """Collect broker summary data for all stocks"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self._broker_categories = load_broker_categories()

    def _parse_number(self, value) -> float:
        """Parse numeric values that may be formatted as strings with commas"""
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

    def get_broker_summary(self, symbol: str, date: datetime = None) -> List[Dict]:
        """
        Get broker summary for a stock

        Args:
            symbol: Stock symbol

        Returns:
            List of broker activity records
        """
        try:
            if date is None:
                date = datetime.now()
            date_str = date.strftime("%Y-%m-%d")
            result = self.api.emiten_broker_summary(
                symbol,
                from_date=date_str,
                to_date=date_str,
                transaction_type="TRANSACTION_TYPE_NET",
                market_board="MARKET_BOARD_ALL",
                investor_type="INVESTOR_TYPE_ALL",
            )
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
                if not broker_list:
                    broker_list = result.get("data", {}).get("data", [])
                if not broker_list:
                    summary = result.get("data", {}).get("broker_summary")
                    if isinstance(summary, dict):
                        broker_map = {}
                        for item in summary.get("brokers_buy", []):
                            code = item.get("netbs_broker_code") or item.get("broker_code")
                            if not code:
                                continue
                            broker_map.setdefault(code, {})
                            broker_map[code].update({
                                "broker_code": code,
                                "buy_value": self._parse_number(item.get("bval")),
                                "buy_volume": self._parse_number(item.get("blotv")),
                                "buy_frequency": abs(int(self._parse_number(item.get("blot")))),
                            })
                        for item in summary.get("brokers_sell", []):
                            code = item.get("netbs_broker_code") or item.get("broker_code")
                            if not code:
                                continue
                            broker_map.setdefault(code, {})
                            sell_value = abs(self._parse_number(item.get("sval")))
                            sell_volume = abs(self._parse_number(item.get("slotv")))
                            sell_freq = abs(int(self._parse_number(item.get("slot"))))
                            broker_map[code].update({
                                "broker_code": code,
                                "sell_value": sell_value,
                                "sell_volume": sell_volume,
                                "sell_frequency": sell_freq,
                            })

                        broker_list = []
                        for code, values in broker_map.items():
                            buy_value = values.get("buy_value", 0.0)
                            sell_value = values.get("sell_value", 0.0)
                            buy_volume = values.get("buy_volume", 0.0)
                            sell_volume = values.get("sell_volume", 0.0)
                            broker_list.append({
                                "broker_code": code,
                                "buy_value": buy_value,
                                "sell_value": sell_value,
                                "net_value": buy_value - sell_value,
                                "buy_volume": buy_volume,
                                "sell_volume": sell_volume,
                                "net_volume": buy_volume - sell_volume,
                                "buy_frequency": values.get("buy_frequency", 0),
                                "sell_frequency": values.get("sell_frequency", 0),
                            })
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
            broker_code = broker_code.strip().upper()
            category_entry = self._broker_categories.get(broker_code, {})
            broker_name = (
                broker.get("broker_name")
                or broker.get("name")
                or broker.get("brokerName")
                or category_entry.get("broker_name")
            )

            parsed.append({
                "date": date.date() if isinstance(date, datetime) else date,
                "broker_code": broker_code,
                "broker_name": broker_name,
                "broker_category": category_entry.get("category") or None,
                "buy_value": self._parse_number(broker.get("buy_value") or broker.get("bval") or broker.get("buyValue")),
                "sell_value": self._parse_number(broker.get("sell_value") or broker.get("sval") or broker.get("sellValue")),
                "net_value": self._parse_number(broker.get("net_value") or broker.get("nval") or broker.get("netValue")),
                "buy_volume": self._parse_number(broker.get("buy_volume") or broker.get("bvol") or broker.get("buyVolume")),
                "sell_volume": self._parse_number(broker.get("sell_volume") or broker.get("svol") or broker.get("sellVolume")),
                "net_volume": self._parse_number(broker.get("net_volume") or broker.get("nvol") or broker.get("netVolume")),
                "buy_frequency": int(self._parse_number(broker.get("buy_frequency") or broker.get("bfreq") or broker.get("buyFreq"))),
                "sell_frequency": int(self._parse_number(broker.get("sell_frequency") or broker.get("sfreq") or broker.get("sellFreq"))),
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
        show_progress: bool = True,
        date: datetime = None
    ) -> Dict[str, int]:
        """
        Collect and save broker summary data for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all active stocks.
            show_progress: Show progress bar
            date: Specific date to collect (defaults to today)

        Returns:
            Dict with collection statistics
        """
        stats = {"success": 0, "failed": 0, "records": 0}
        target_date = date or datetime.now()

        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(
            symbols,
            desc=f"Collecting broker summaries ({target_date.date()})"
        ) if show_progress else symbols

        for symbol in iterator:
            try:
                broker_data = self.get_broker_summary(symbol, date=target_date)

                if broker_data:
                    parsed = self.parse_broker_data(broker_data, target_date)

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

        print(
            f"\nBroker collection complete ({target_date.date()}): "
            f"{stats['success']} stocks, {stats['records']} records"
        )
        return stats

    def collect_today(self) -> Dict[str, int]:
        """Collect today's broker summary for all stocks"""
        return self.collect_and_save()

    def collect_range(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        """Collect broker summaries for a date range (trading days only)."""
        from utils.holidays import is_trading_day

        def _to_date(value: str) -> datetime:
            return datetime.strptime(value, "%Y-%m-%d")

        start = _to_date(start_date)
        end = _to_date(end_date)
        if end < start:
            raise ValueError("end_date must be >= start_date")

        dates = []
        current = start
        while current <= end:
            if is_trading_day(current):
                dates.append(current)
            current = current + timedelta(days=1)

        iterator = tqdm(dates, desc="Collecting broker summaries (range)") if show_progress else dates
        results = {}
        for day in iterator:
            date_str = day.strftime("%Y-%m-%d")
            results[date_str] = self.collect_and_save(
                symbols=symbols,
                show_progress=False,
                date=day,
            )
        return results


if __name__ == "__main__":
    from database import init_db
    init_db()

    collector = BrokerSummaryCollector()
    # Test with single stock
    result = collector.get_broker_summary("BBCA")
    print(f"BBCA broker data: {len(result) if result else 0} brokers")
    if result:
        print(f"Sample: {result[0] if result else 'N/A'}")
