"""
Insider Trading Collector - Collect insider transaction data
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock
)
from database.models import InsiderTrade
from config import DATASAHAM_API_KEY, API_RATE_LIMIT


class InsiderTradeCollector:
    """Collect insider trading data for all stocks"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)

    def get_insider_data(self, symbol: str) -> List[Dict]:
        """
        Get insider trading data for a stock

        Args:
            symbol: Stock symbol

        Returns:
            List of insider trading records
        """
        try:
            result = self.api.emiten_insider(symbol)
            # Handle various response structures
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                insider_list = result.get("insider_list", [])
                if not insider_list:
                    insider_list = result.get("data", {}).get("insider_list", [])
                if not insider_list:
                    insider_list = result.get("insiders", [])
                if not insider_list:
                    insider_list = result.get("transactions", [])
                if not insider_list and "data" in result:
                    data = result.get("data")
                    if isinstance(data, list):
                        insider_list = data
                return insider_list
            return []
        except Exception as e:
            print(f"Error getting insider data for {symbol}: {e}")
            return []

    def parse_date(self, date_value) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value.date()
        if isinstance(date_value, str):
            for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
        return None

    def parse_insider_data(self, insider_data: List[Dict]) -> List[Dict]:
        """Parse insider data into standardized format"""
        parsed = []
        for insider in insider_data:
            # Get insider name
            insider_name = (
                insider.get("insider_name") or
                insider.get("name") or
                insider.get("insiderName") or
                insider.get("nama")
            )
            if not insider_name:
                continue

            # Parse transaction date
            tx_date = self.parse_date(
                insider.get("transaction_date") or
                insider.get("date") or
                insider.get("transactionDate") or
                insider.get("tanggal")
            )
            if not tx_date:
                tx_date = datetime.now().date()

            # Parse transaction type
            tx_type = (
                insider.get("transaction_type") or
                insider.get("type") or
                insider.get("transactionType") or
                insider.get("tipe") or
                ""
            ).lower()
            if tx_type not in ("buy", "sell"):
                # Try to infer from other fields
                if "beli" in tx_type or "buy" in tx_type or "acquire" in tx_type:
                    tx_type = "buy"
                elif "jual" in tx_type or "sell" in tx_type or "dispose" in tx_type:
                    tx_type = "sell"
                else:
                    tx_type = "buy"  # Default

            parsed.append({
                "insider_name": insider_name,
                "position": insider.get("position") or insider.get("jabatan") or insider.get("role"),
                "relationship_type": insider.get("relationship") or insider.get("relationship_type"),
                "transaction_type": tx_type,
                "transaction_date": tx_date,
                "shares": insider.get("shares") or insider.get("volume") or insider.get("saham") or 0,
                "price": insider.get("price") or insider.get("harga") or 0,
                "value": insider.get("value") or insider.get("nilai") or 0,
                "shares_after": insider.get("shares_after") or insider.get("ownership_after") or insider.get("saldo"),
                "announcement_date": self.parse_date(insider.get("announcement_date")),
            })
        return parsed

    def _insert_new_insider_trades(self, session, stock_id: int, insider_data: list) -> int:
        """Insert new insider trades (avoid duplicates based on key fields)"""
        new_count = 0
        for data in insider_data:
            # Check for existing record with same key fields
            existing = session.query(InsiderTrade).filter(
                InsiderTrade.stock_id == stock_id,
                InsiderTrade.insider_name == data["insider_name"],
                InsiderTrade.transaction_date == data["transaction_date"],
                InsiderTrade.transaction_type == data["transaction_type"],
                InsiderTrade.shares == data["shares"]
            ).first()

            if not existing:
                record = InsiderTrade(stock_id=stock_id, **data)
                session.add(record)
                new_count += 1

        return new_count

    def collect_and_save(
        self,
        symbols: List[str] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Collect and save insider trading data

        Args:
            symbols: List of stock symbols. If None, uses all active stocks.
            show_progress: Show progress bar

        Returns:
            Dict with collection statistics
        """
        stats = {"success": 0, "failed": 0, "records": 0, "new_records": 0}

        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(symbols, desc="Collecting insider trades") if show_progress else symbols

        for symbol in iterator:
            try:
                insider_data = self.get_insider_data(symbol)

                if insider_data:
                    parsed = self.parse_insider_data(insider_data)

                    if parsed:
                        with session_scope() as session:
                            stock = get_stock_by_symbol(session, symbol)
                            if stock:
                                new_count = self._insert_new_insider_trades(session, stock.id, parsed)
                                stats["records"] += len(parsed)
                                stats["new_records"] += new_count
                                stats["success"] += 1
                            else:
                                stats["failed"] += 1
                    else:
                        stats["success"] += 1  # No insider data is valid
                else:
                    # No insider data is common - not a failure
                    stats["success"] += 1

                time.sleep(API_RATE_LIMIT)

            except Exception as e:
                print(f"Error collecting insider data for {symbol}: {e}")
                stats["failed"] += 1
                continue

        print(f"\nInsider collection complete: {stats['success']} stocks, {stats['new_records']} new records")
        return stats

    def collect_today(self) -> Dict[str, int]:
        """Collect insider trading data for all stocks"""
        return self.collect_and_save()


if __name__ == "__main__":
    from database import init_db
    init_db()

    collector = InsiderTradeCollector()
    # Test with single stock
    result = collector.get_insider_data("BBCA")
    print(f"BBCA insider data: {len(result) if result else 0} transactions")
    if result:
        print(f"Sample: {result[0] if result else 'N/A'}")
