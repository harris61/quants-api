"""
Stock List Collector - Get all stock symbols from IDX
"""

import time
import re
from typing import List, Dict, Any
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import session_scope, get_or_create_stock, Stock
from config import DATASAHAM_API_KEY, API_RATE_LIMIT, EQUITY_SYMBOL_REGEX


class StockListCollector:
    """Collect and update list of all stocks from IDX"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def is_equity_symbol(self, symbol: str) -> bool:
        """Return True for normal equity tickers"""
        if not symbol:
            return False
        return bool(self._equity_pattern.match(symbol.upper()))

    def get_all_sectors(self) -> List[Dict[str, Any]]:
        """Get list of all sectors"""
        result = self.api.sectors()
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict):
                return data.get("data", [])
            if isinstance(data, list):
                return data
        return result if isinstance(result, list) else []

    def get_stocks_by_sector(self, sector_id: str) -> List[Dict[str, Any]]:
        """Get all stocks in a specific sector"""
        subsectors = self.get_subsectors_by_sector(sector_id)
        all_companies = []
        for subsector in subsectors:
            subsector_id = subsector.get("id")
            companies = self.get_companies_by_subsector(sector_id, subsector_id)
            all_companies.extend(companies)
            self.api.smart_sleep()
        return all_companies

    def get_subsectors_by_sector(self, sector_id: str) -> List[Dict[str, Any]]:
        """Get subsectors in a specific sector"""
        result = self.api.sector_subsectors(sector_id)
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict):
                return data.get("data", [])
            if isinstance(data, list):
                return data
        return result if isinstance(result, list) else []

    def get_companies_by_subsector(self, sector_id: str, subsector_id: str) -> List[Dict[str, Any]]:
        """Get companies in a specific subsector"""
        result = self.api.sector_subsector_companies(sector_id, subsector_id)
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict):
                return data.get("data", [])
            if isinstance(data, list):
                return data
        return result if isinstance(result, list) else []

    def collect_all_stocks(self) -> List[Dict[str, Any]]:
        """
        Collect all stocks from all sectors

        Returns:
            List of stock dictionaries with symbol, name, sector info
        """
        all_stocks = []
        seen_symbols = set()

        # Get all sectors
        sectors = self.get_all_sectors()
        print(f"Found {len(sectors)} sectors")

        for sector in tqdm(sectors, desc="Collecting stocks by sector"):
            sector_id = sector.get("id")
            sector_name = sector.get("name")

            try:
                stocks = self.get_stocks_by_sector(sector_id)

                for stock in stocks:
                    symbol = stock.get("code") or stock.get("symbol")
                    if symbol and symbol not in seen_symbols and self.is_equity_symbol(symbol):
                        seen_symbols.add(symbol)
                        all_stocks.append({
                            "symbol": symbol,
                            "name": stock.get("name"),
                            "sector_id": sector_id,
                            "sector_name": sector_name,
                        })

                # Rate limiting
                self.api.smart_sleep()

            except Exception as e:
                print(f"Error collecting stocks from sector {sector_id}: {e}")
                continue

        print(f"Total unique stocks collected: {len(all_stocks)}")
        return all_stocks

    def save_to_database(self, stocks: List[Dict[str, Any]] = None) -> int:
        """
        Save stock list to database

        Args:
            stocks: List of stock dicts. If None, will collect from API.

        Returns:
            Number of stocks saved/updated
        """
        if stocks is None:
            stocks = self.collect_all_stocks()

        count = 0
        with session_scope() as session:
            for stock_data in tqdm(stocks, desc="Saving to database"):
                try:
                    stock = get_or_create_stock(
                        session,
                        symbol=stock_data["symbol"],
                        name=stock_data.get("name"),
                        sector_id=stock_data.get("sector_id"),
                        sector_name=stock_data.get("sector_name"),
                    )
                    # Update existing stock info
                    if stock_data.get("name"):
                        stock.name = stock_data["name"]
                    if stock_data.get("sector_id"):
                        stock.sector_id = stock_data["sector_id"]
                    if stock_data.get("sector_name"):
                        stock.sector_name = stock_data["sector_name"]

                    count += 1
                except Exception as e:
                    print(f"Error saving stock {stock_data.get('symbol')}: {e}")
                    continue

        print(f"Saved {count} stocks to database")
        return count

    def get_active_symbols(self) -> List[str]:
        """Get list of active stock symbols from database"""
        with session_scope() as session:
            stocks = session.query(Stock).filter(Stock.is_active == True).all()
            return [s.symbol for s in stocks]


def collect_stocks():
    """CLI function to collect all stocks"""
    from database import init_db

    # Initialize database
    init_db()

    # Collect and save stocks
    collector = StockListCollector()
    stocks = collector.collect_all_stocks()
    collector.save_to_database(stocks)


if __name__ == "__main__":
    collect_stocks()
