"""
Daily Data Collector - Collect daily OHLCV and trading data
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock, DailyPrice,
    bulk_upsert_daily_prices
)
import re

from config import DATASAHAM_API_KEY, API_RATE_LIMIT, EQUITY_SYMBOL_REGEX


class DailyDataCollector:
    """Collect daily OHLCV and trading data for all stocks"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def _is_equity_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        return bool(self._equity_pattern.match(symbol.upper()))

    def get_chart_data(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """
        Get daily OHLCV data for a stock

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD) - more recent
            to_date: End date (YYYY-MM-DD) - older

        Returns:
            List of daily price data
        """
        try:
            result = self.api.chart_daily(symbol, from_date, to_date)
            chartbit = result.get("data", {}).get("chartbit", [])
            return chartbit
        except Exception as e:
            print(f"Error getting chart data for {symbol}: {e}")
            return []

    def get_top_gainers(self) -> List[Dict]:
        """Get today's top gainers with foreign flow data"""
        try:
            result = self.api.top_gainer()
            return result.get("data", {}).get("mover_list", [])
        except Exception as e:
            print(f"Error getting top gainers: {e}")
            return []

    def get_top_losers(self) -> List[Dict]:
        """Get today's top losers with foreign flow data"""
        try:
            result = self.api.top_loser()
            return result.get("data", {}).get("mover_list", [])
        except Exception as e:
            print(f"Error getting top losers: {e}")
            return []

    def get_net_foreign_buy(self) -> List[Dict]:
        """Get stocks with highest net foreign buy"""
        try:
            result = self.api.net_foreign_buy()
            return result.get("data", {}).get("mover_list", [])
        except Exception as e:
            print(f"Error getting net foreign buy: {e}")
            return []

    def get_net_foreign_sell(self) -> List[Dict]:
        """Get stocks with highest net foreign sell"""
        try:
            result = self.api.net_foreign_sell()
            return result.get("data", {}).get("mover_list", [])
        except Exception as e:
            print(f"Error getting net foreign sell: {e}")
            return []

    def parse_chart_data(self, chart_data: List[Dict]) -> List[Dict]:
        """Parse chart data into standardized format"""
        parsed = []
        for candle in chart_data:
            parsed.append({
                "date": candle.get("date"),
                "open": candle.get("open"),
                "high": candle.get("high"),
                "low": candle.get("low"),
                "close": candle.get("close"),
                "volume": candle.get("volume"),
                "value": candle.get("value"),
                "frequency": candle.get("frequency"),
                "foreign_buy": candle.get("foreignBuy") or candle.get("foreign_buy"),
                "foreign_sell": candle.get("foreignSell") or candle.get("foreign_sell"),
                "foreign_net": candle.get("foreignNet") or candle.get("foreign_net"),
                "change": candle.get("change"),
                "change_percent": candle.get("percent") or candle.get("change_percent"),
            })
        return parsed

    def collect_stock_data(
        self,
        symbol: str,
        days: int = 30,
        from_date: str = None,
        to_date: str = None
    ) -> List[Dict]:
        """
        Collect daily data for a single stock

        Args:
            symbol: Stock symbol
            days: Number of days to collect (if from/to_date not specified)
            from_date: Optional start date
            to_date: Optional end date

        Returns:
            List of daily price records
        """
        if not self._is_equity_symbol(symbol):
            return []
        if from_date is None:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if to_date is None:
            to_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        chart_data = self.get_chart_data(symbol, from_date, to_date)
        return self.parse_chart_data(chart_data)

    def collect_and_save(
        self,
        symbols: List[str] = None,
        days: int = 30,
        from_date: str = None,
        to_date: str = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Collect and save daily data for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all active stocks from DB.
            days: Number of days to collect
            from_date: Optional start date
            to_date: Optional end date
            show_progress: Show progress bar

        Returns:
            Dict with collection statistics
        """
        stats = {"success": 0, "failed": 0, "records": 0}

        # Get symbols from database if not provided
        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks if self._is_equity_symbol(s.symbol)]
        else:
            symbols = [s for s in symbols if self._is_equity_symbol(s)]

        if not symbols:
            print("No symbols to collect data for!")
            return stats

        iterator = tqdm(symbols, desc="Collecting daily data") if show_progress else symbols

        for symbol in iterator:
            try:
                # Collect data
                price_data = self.collect_stock_data(
                    symbol, days=days, from_date=from_date, to_date=to_date
                )

                if price_data:
                    # Save to database
                    with session_scope() as session:
                        stock = get_stock_by_symbol(session, symbol)
                        if stock:
                            bulk_upsert_daily_prices(session, stock.id, price_data)
                            stats["records"] += len(price_data)
                            stats["success"] += 1
                        else:
                            print(f"Stock {symbol} not found in database")
                            stats["failed"] += 1
                else:
                    stats["failed"] += 1

                # Rate limiting
                time.sleep(API_RATE_LIMIT)

            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
                stats["failed"] += 1
                continue

        print(f"\nCollection complete: {stats['success']} stocks, {stats['records']} records")
        return stats

    def collect_today(self) -> Dict[str, int]:
        """Collect today's data for all stocks"""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return self.collect_and_save(from_date=today, to_date=yesterday)


def collect_daily():
    """CLI function to collect daily data"""
    from database import init_db

    # Initialize database
    init_db()

    # Collect today's data
    collector = DailyDataCollector()
    collector.collect_today()


if __name__ == "__main__":
    collect_daily()
