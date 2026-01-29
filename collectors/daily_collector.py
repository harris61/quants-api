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

    def collect_foreign_flow(self, date: str = None) -> Dict[str, int]:
        """
        Collect foreign flow data from net foreign endpoints and update daily_prices.

        The chart_daily API doesn't return foreign data, but the net foreign
        endpoints do. This method fetches those lists and updates the
        corresponding daily_prices records.

        Args:
            date: Date to update (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict with collection statistics
        """
        from database import session_scope, get_stock_by_symbol, DailyPrice

        stats = {"updated": 0, "not_found": 0}

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Fetch foreign buy list
        foreign_buy_list = self.get_net_foreign_buy()
        # Fetch foreign sell list
        foreign_sell_list = self.get_net_foreign_sell()

        # Combine into a dict by symbol
        foreign_data = {}

        for item in foreign_buy_list:
            symbol = item.get("stock_detail", {}).get("code")
            if not symbol or not self._is_equity_symbol(symbol):
                continue
            foreign_buy = item.get("net_foreign_buy", {}).get("raw", 0) or 0
            foreign_sell = item.get("net_foreign_sell", {}).get("raw", 0) or 0
            foreign_data[symbol] = {
                "foreign_buy": foreign_buy,
                "foreign_sell": foreign_sell,
                "foreign_net": foreign_buy - foreign_sell
            }

        for item in foreign_sell_list:
            symbol = item.get("stock_detail", {}).get("code")
            if not symbol or not self._is_equity_symbol(symbol):
                continue
            if symbol not in foreign_data:
                foreign_buy = item.get("net_foreign_buy", {}).get("raw", 0) or 0
                foreign_sell = item.get("net_foreign_sell", {}).get("raw", 0) or 0
                foreign_data[symbol] = {
                    "foreign_buy": foreign_buy,
                    "foreign_sell": foreign_sell,
                    "foreign_net": foreign_buy - foreign_sell
                }

        # Update daily_prices records
        with session_scope() as session:
            for symbol, data in foreign_data.items():
                stock = get_stock_by_symbol(session, symbol)
                if not stock:
                    stats["not_found"] += 1
                    continue

                # Find the daily_price record for this stock and date
                price_record = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == stock.id,
                    DailyPrice.date == date
                ).first()

                if price_record:
                    price_record.foreign_buy = data["foreign_buy"]
                    price_record.foreign_sell = data["foreign_sell"]
                    price_record.foreign_net = data["foreign_net"]
                    stats["updated"] += 1
                else:
                    stats["not_found"] += 1

        print(f"Foreign flow collection: {stats['updated']} updated, {stats['not_found']} not found")
        return stats

    def _fetch_broker_daily_foreign(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        chunk_days: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fetch broker summary and aggregate foreign buy/sell by date for a symbol."""
        def _merge_into(target: Dict[str, Dict[str, float]], source: Dict[str, Dict[str, float]]) -> None:
            for date, data in source.items():
                if date not in target:
                    target[date] = {"buy": 0.0, "sell": 0.0}
                target[date]["buy"] += data.get("buy", 0.0)
                target[date]["sell"] += data.get("sell", 0.0)

        def _fetch_range(range_start: str, range_end: str) -> Dict[str, Dict[str, float]]:
            result = self.api._request(f"market-detector/broker-summary/{symbol}", {
                "from": range_start,
                "to": range_end,
                "transactionType": "TRANSACTION_TYPE_NET",
                "investorType": "INVESTOR_TYPE_FOREIGN",
                "limit": 100,
            })

            broker_data = {}
            if isinstance(result, dict):
                broker_data = result.get("broker_summary") or result.get("brokerSummary") or {}
                if not broker_data and isinstance(result.get("data"), dict):
                    broker_data = (
                        result["data"].get("broker_summary")
                        or result["data"].get("brokerSummary")
                        or {}
                    )
            brokers_buy = broker_data.get("brokers_buy") or broker_data.get("brokersBuy") or []
            brokers_sell = broker_data.get("brokers_sell") or broker_data.get("brokersSell") or []

            daily_foreign: Dict[str, Dict[str, float]] = {}
            for broker in brokers_buy:
                date_str = broker.get("netbs_date") or broker.get("date") or broker.get("trade_date") or ""
                if not date_str:
                    continue
                date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                if date not in daily_foreign:
                    daily_foreign[date] = {"buy": 0.0, "sell": 0.0}
                daily_foreign[date]["buy"] += float(broker.get("bval", 0) or 0)

            for broker in brokers_sell:
                date_str = broker.get("netbs_date") or broker.get("date") or broker.get("trade_date") or ""
                if not date_str:
                    continue
                date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                if date not in daily_foreign:
                    daily_foreign[date] = {"buy": 0.0, "sell": 0.0}
                daily_foreign[date]["sell"] += float(broker.get("sval", 0) or 0)

            return daily_foreign

        if not chunk_days or chunk_days <= 0:
            return _fetch_range(start_date, end_date)

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return _fetch_range(start_date, end_date)

        merged: Dict[str, Dict[str, float]] = {}
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            chunk_data = _fetch_range(current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
            _merge_into(merged, chunk_data)
            current = chunk_end + timedelta(days=1)
        return merged

    def backfill_foreign_flow(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Backfill historical foreign flow data using broker_summary API.

        This is slow because it requires one API call per stock.
        Use for historical backfill only.

        Args:
            start_date: Start date (YYYY-MM-DD) - older date
            end_date: End date (YYYY-MM-DD) - more recent date
            symbols: List of symbols to backfill. If None, uses all active stocks.
            show_progress: Show progress bar

        Returns:
            Dict with backfill statistics
        """
        from database import session_scope, get_stock_by_symbol, Stock, DailyPrice

        stats = {"stocks_processed": 0, "records_updated": 0, "errors": 0}

        # Get symbols from database if not provided
        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks if self._is_equity_symbol(s.symbol)]

        print(f"Backfilling foreign flow for {len(symbols)} stocks from {start_date} to {end_date}")
        print("This may take a while...")

        iterator = tqdm(symbols, desc="Backfilling foreign flow") if show_progress else symbols

        for symbol in iterator:
            try:
                daily_foreign = self._fetch_broker_daily_foreign(symbol, start_date, end_date)

                # Update daily_prices records
                with session_scope() as session:
                    stock = get_stock_by_symbol(session, symbol)
                    if not stock:
                        continue

                    for date, data in daily_foreign.items():
                        price_record = session.query(DailyPrice).filter(
                            DailyPrice.stock_id == stock.id,
                            DailyPrice.date == date
                        ).first()

                        if price_record:
                            price_record.foreign_buy = data["buy"]
                            price_record.foreign_sell = data["sell"]
                            price_record.foreign_net = data["buy"] - data["sell"]
                            stats["records_updated"] += 1

                stats["stocks_processed"] += 1
                time.sleep(API_RATE_LIMIT)

            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error for {symbol}: {e}")
                stats["errors"] += 1
                continue

        print(f"\nBackfill complete: {stats['stocks_processed']} stocks, {stats['records_updated']} records updated, {stats['errors']} errors")
        return stats

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
