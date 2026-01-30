"""
Market Cap Collector - Collect current market cap from Datasaham sector/subsector companies
"""

import time
import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm

from datasaham import DatasahamAPI
from database import session_scope, get_stock_by_symbol, Stock
from database.models import MarketCapHistory
from config import DATASAHAM_API_KEY, API_RATE_LIMIT, EQUITY_SYMBOL_REGEX


class MarketCapCollector:
    """Collect current market cap for all active stocks"""

    _MARKET_CAP_KEYS = ("marketcap", "market_cap", "marketCap")

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def _is_equity_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        return bool(self._equity_pattern.match(symbol.upper()))

    def _coerce_number(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            raw = value.get("raw")
            if raw is not None:
                return self._coerce_number(raw)
            val = value.get("value")
            if val is not None:
                return self._coerce_number(val)
        if isinstance(value, str):
            cleaned = value.replace(",", "").replace(".", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _extract_market_cap(self, data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        if not isinstance(data, dict):
            return None, None, None
        for key in self._MARKET_CAP_KEYS:
            if key in data:
                val = data.get(key)
                numeric = self._coerce_number(val)
                formatted = None
                if isinstance(val, str):
                    formatted = val
                currency = data.get("currency") or data.get("market_cap_currency")
                return numeric, formatted, currency
        return None, None, None

    def _extract_list(self, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict):
                return data.get("data", [])
            if isinstance(data, list):
                return data
        return result if isinstance(result, list) else []

    def _get_all_companies(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        companies: List[Dict[str, Any]] = []
        sectors = self._extract_list(self.api.sectors())
        iterator = tqdm(sectors, desc="Collecting sectors") if show_progress else sectors
        for sector in iterator:
            sector_id = sector.get("id")
            if not sector_id:
                continue
            subsectors = self._extract_list(self.api.sector_subsectors(sector_id))
            for subsector in subsectors:
                subsector_id = subsector.get("id")
                if not subsector_id:
                    continue
                result = self.api.sector_subsector_companies(sector_id, subsector_id)
                companies.extend(self._extract_list(result))
                time.sleep(API_RATE_LIMIT)
        return companies

    def _upsert_history(
        self,
        session,
        stock_id: int,
        snapshot_date: date,
        market_cap: Optional[float],
        formatted: Optional[str],
        currency: Optional[str],
    ) -> None:
        """Upsert market cap snapshot for a stock/date."""
        from sqlalchemy.dialects.sqlite import insert

        record = {
            "stock_id": stock_id,
            "date": snapshot_date,
            "market_cap": market_cap,
            "market_cap_formatted": formatted,
            "market_cap_currency": currency,
        }
        stmt = insert(MarketCapHistory).values(**record)
        update_fields = {
            "market_cap": stmt.excluded.market_cap,
            "market_cap_formatted": stmt.excluded.market_cap_formatted,
            "market_cap_currency": stmt.excluded.market_cap_currency,
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "date"],
            set_=update_fields
        )
        session.execute(stmt)

    def collect_and_save(
        self,
        symbols: List[str] = None,
        show_progress: bool = True,
        snapshot_date: Optional[date] = None,
        save_history: bool = False,
    ) -> Dict[str, int]:
        stats = {"success": 0, "missing": 0, "not_found": 0, "failed": 0}
        snap_date = snapshot_date or datetime.utcnow().date()

        if symbols:
            symbols = [s for s in symbols if self._is_equity_symbol(s)]

        companies = self._get_all_companies(show_progress=show_progress)
        if symbols:
            wanted = {s.upper() for s in symbols}
            companies = [c for c in companies if (c.get("symbol") or c.get("code") or "").upper() in wanted]

        if not companies:
            print("No companies found to collect market cap for!")
            return stats

        iterator = tqdm(companies, desc="Saving market cap") if show_progress else companies
        for company in iterator:
            try:
                symbol = company.get("symbol") or company.get("code")
                if not self._is_equity_symbol(symbol):
                    continue
                market_cap, formatted, currency = self._extract_market_cap(company)
                if market_cap is None and formatted is None:
                    stats["missing"] += 1
                    continue
                with session_scope() as session:
                    stock = get_stock_by_symbol(session, symbol)
                    if not stock:
                        stats["not_found"] += 1
                        continue
                    stock.market_cap = market_cap
                    stock.market_cap_formatted = formatted
                    stock.market_cap_currency = currency
                    stock.market_cap_updated_at = datetime.utcnow()
                    if save_history:
                        self._upsert_history(
                            session,
                            stock.id,
                            snap_date,
                            market_cap,
                            formatted,
                            currency,
                        )
                    stats["success"] += 1
            except Exception as exc:
                if show_progress:
                    tqdm.write(f"Error saving market cap for {company.get('symbol')}: {exc}")
                stats["failed"] += 1
                continue

        print(
            "Market cap collection complete: "
            f"{stats['success']} updated, {stats['missing']} missing, "
            f"{stats['not_found']} not found, {stats['failed']} failed"
        )
        return stats


def collect_market_cap():
    """CLI function to collect market cap data"""
    from database import init_db

    init_db()
    collector = MarketCapCollector()
    collector.collect_and_save()


if __name__ == "__main__":
    collect_market_cap()
