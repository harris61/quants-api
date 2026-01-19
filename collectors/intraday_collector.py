"""
Intraday OHLCV Collector - Collect hourly price data
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock
)
from database.models import IntradayPrice
from config import DATASAHAM_API_KEY, API_RATE_LIMIT


class IntradayCollector:
    """Collect intraday (hourly) OHLCV data for all stocks"""

    def __init__(self, api_key: str = None, interval: str = "1h"):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self.interval = interval  # Default 1h interval

    def get_intraday_data(
        self,
        symbol: str,
        from_date: str,
        to_date: str
    ) -> List[Dict]:
        """
        Get intraday OHLCV data for a stock

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD) - more recent
            to_date: End date (YYYY-MM-DD) - older

        Returns:
            List of intraday price records
        """
        try:
            result = self.api.chart_intraday(symbol, self.interval, from_date, to_date)
            # Handle response structure similar to daily chart
            if isinstance(result, dict):
                chartbit = result.get("data", {}).get("chartbit", [])
                if not chartbit:
                    chartbit = result.get("chartbit", [])
                if not chartbit and "data" in result:
                    data = result.get("data")
                    if isinstance(data, list):
                        chartbit = data
                return chartbit
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            print(f"Error getting intraday data for {symbol}: {e}")
            return []

    def parse_datetime(self, dt_value) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if dt_value is None:
            return None
        if isinstance(dt_value, datetime):
            return dt_value
        if isinstance(dt_value, str):
            # Try various datetime formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(dt_value, fmt)
                except ValueError:
                    continue
            # Try parsing as date only (set time to 00:00)
            try:
                return datetime.strptime(dt_value, "%Y-%m-%d")
            except ValueError:
                pass
        return None

    def parse_intraday_data(self, intraday_data: List[Dict]) -> List[Dict]:
        """Parse intraday data into standardized format"""
        parsed = []
        for candle in intraday_data:
            # Parse datetime
            dt = self.parse_datetime(
                candle.get("datetime") or
                candle.get("date") or
                candle.get("timestamp")
            )
            if not dt:
                continue

            parsed.append({
                "datetime": dt,
                "date": dt.date(),
                "hour": dt.hour,
                "open": candle.get("open") or 0,
                "high": candle.get("high") or 0,
                "low": candle.get("low") or 0,
                "close": candle.get("close") or 0,
                "volume": candle.get("volume") or 0,
                "value": candle.get("value") or 0,
            })
        return parsed

    def _bulk_upsert_intraday_prices(self, session, stock_id: int, price_data: list):
        """Bulk insert or update intraday price records"""
        from sqlalchemy.dialects.sqlite import insert

        for data in price_data:
            record = {
                "stock_id": stock_id,
                **data
            }
            stmt = insert(IntradayPrice).values(**record)
            update_fields = {k: stmt.excluded[k] for k in data.keys() if k != "datetime"}
            stmt = stmt.on_conflict_do_update(
                index_elements=["stock_id", "datetime"],
                set_=update_fields
            )
            session.execute(stmt)

    def collect_stock_intraday(
        self,
        symbol: str,
        days: int = 5,
        from_date: str = None,
        to_date: str = None
    ) -> List[Dict]:
        """
        Collect intraday data for a single stock

        Args:
            symbol: Stock symbol
            days: Number of days to collect
            from_date: Optional start date (more recent)
            to_date: Optional end date (older)

        Returns:
            List of intraday price records
        """
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        intraday_data = self.get_intraday_data(symbol, from_date, to_date)
        return self.parse_intraday_data(intraday_data)

    def collect_and_save(
        self,
        symbols: List[str] = None,
        days: int = 5,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Collect and save intraday data for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all active stocks.
            days: Number of days of intraday data
            show_progress: Show progress bar

        Returns:
            Dict with collection statistics
        """
        stats = {"success": 0, "failed": 0, "records": 0}

        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(symbols, desc="Collecting intraday data") if show_progress else symbols

        for symbol in iterator:
            try:
                intraday_data = self.collect_stock_intraday(symbol, days=days)

                if intraday_data:
                    with session_scope() as session:
                        stock = get_stock_by_symbol(session, symbol)
                        if stock:
                            self._bulk_upsert_intraday_prices(session, stock.id, intraday_data)
                            stats["records"] += len(intraday_data)
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1
                else:
                    stats["success"] += 1  # No intraday data is valid

                time.sleep(API_RATE_LIMIT)

            except Exception as e:
                print(f"Error collecting intraday data for {symbol}: {e}")
                stats["failed"] += 1
                continue

        print(f"\nIntraday collection complete: {stats['success']} stocks, {stats['records']} records")
        return stats

    def collect_today(self) -> Dict[str, int]:
        """Collect today's intraday data for all stocks"""
        return self.collect_and_save(days=1)


if __name__ == "__main__":
    from database import init_db
    init_db()

    collector = IntradayCollector()
    # Test with single stock
    data = collector.collect_stock_intraday("BBCA", days=1)
    print(f"BBCA intraday data: {len(data)} candles")
    if data:
        print(f"Sample: {data[0] if data else 'N/A'}")
