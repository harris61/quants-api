"""
Historical Data Loader - Batch load historical data for all stocks
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import (
    session_scope, get_stock_by_symbol, Stock, DailyPrice,
    bulk_upsert_daily_prices, init_db
)
from config import DATASAHAM_API_KEY, API_RATE_LIMIT, HISTORICAL_DAYS
from collectors.daily_collector import DailyDataCollector


class HistoricalDataLoader:
    """Load historical data for all stocks"""

    def __init__(self, api_key: str = None, days: int = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self.collector = DailyDataCollector(api_key or DATASAHAM_API_KEY)
        self.days = days or HISTORICAL_DAYS

    def get_symbols_from_db(self) -> List[str]:
        """Get all active stock symbols from database"""
        with session_scope() as session:
            stocks = session.query(Stock).filter(Stock.is_active == True).all()
            return [s.symbol for s in stocks if self.collector._is_equity_symbol(s.symbol)]

    def get_last_data_date(self, symbol: str) -> Optional[datetime]:
        """Get the last date we have data for a stock"""
        with session_scope() as session:
            stock = get_stock_by_symbol(session, symbol)
            if stock:
                last_price = session.query(DailyPrice)\
                    .filter(DailyPrice.stock_id == stock.id)\
                    .order_by(DailyPrice.date.desc())\
                    .first()
                if last_price:
                    return last_price.date
        return None

    def load_historical_data(
        self,
        symbols: List[str] = None,
        days: int = None,
        resume: bool = True,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Load historical data for all stocks

        Args:
            symbols: List of stock symbols. If None, uses all from DB.
            days: Number of days to load. If None, uses config default.
            resume: If True, only load data from last known date.
            show_progress: Show progress bar.

        Returns:
            Statistics dict
        """
        stats = {"success": 0, "failed": 0, "records": 0, "skipped": 0}

        if symbols is None:
            symbols = self.get_symbols_from_db()

        if not symbols:
            print("No symbols found! Run stock list collection first.")
            return stats

        days = days or self.days
        today = datetime.now()

        print(f"Loading {days} days of historical data for {len(symbols)} stocks...")

        iterator = tqdm(symbols, desc="Loading historical data") if show_progress else symbols

        for symbol in iterator:
            try:
                # Determine date range
                from_date = today.strftime("%Y-%m-%d")

                if resume:
                    last_date = self.get_last_data_date(symbol)
                    if last_date:
                        # Only load data after last known date
                        days_diff = (today.date() - last_date).days
                        if days_diff <= 1:
                            stats["skipped"] += 1
                            continue
                        to_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    else:
                        to_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
                else:
                    to_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")

                # Collect data
                price_data = self.collector.collect_stock_data(
                    symbol,
                    from_date=from_date,
                    to_date=to_date
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
                            stats["failed"] += 1
                else:
                    stats["failed"] += 1

                self.api.smart_sleep()

            except Exception as e:
                print(f"\nError loading data for {symbol}: {e}")
                stats["failed"] += 1
                continue

        print(f"\nHistorical load complete:")
        print(f"  Success: {stats['success']} stocks")
        print(f"  Failed: {stats['failed']} stocks")
        print(f"  Skipped: {stats['skipped']} stocks (up to date)")
        print(f"  Total records: {stats['records']}")

        return stats

    def verify_data(self) -> Dict[str, any]:
        """Verify loaded data integrity"""
        with session_scope() as session:
            total_stocks = session.query(Stock).count()
            active_stocks = session.query(Stock).filter(Stock.is_active == True).count()
            total_prices = session.query(DailyPrice).count()

            # Get date range
            min_date = session.query(DailyPrice.date)\
                .order_by(DailyPrice.date.asc()).first()
            max_date = session.query(DailyPrice.date)\
                .order_by(DailyPrice.date.desc()).first()

            # Stocks with data
            stocks_with_data = session.query(DailyPrice.stock_id)\
                .distinct().count()

            return {
                "total_stocks": total_stocks,
                "active_stocks": active_stocks,
                "stocks_with_data": stocks_with_data,
                "total_price_records": total_prices,
                "date_range": {
                    "from": min_date[0].strftime("%Y-%m-%d") if min_date else None,
                    "to": max_date[0].strftime("%Y-%m-%d") if max_date else None,
                }
            }


def load_historical():
    """CLI function to load historical data"""
    import argparse

    parser = argparse.ArgumentParser(description="Load historical stock data")
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS,
                        help=f"Number of days to load (default: {HISTORICAL_DAYS})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't resume from last known date")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing data")
    args = parser.parse_args()

    # Initialize database
    init_db()

    loader = HistoricalDataLoader(days=args.days)

    if args.verify:
        print("\nData Verification:")
        print("-" * 40)
        verification = loader.verify_data()
        for key, value in verification.items():
            print(f"  {key}: {value}")
        return

    # Load historical data
    loader.load_historical_data(resume=not args.no_resume)

    # Verify after loading
    print("\nData Verification:")
    print("-" * 40)
    verification = loader.verify_data()
    for key, value in verification.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    load_historical()
