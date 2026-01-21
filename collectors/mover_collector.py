"""
Market Movers Collector - Collect top value/volume/frequency/gainer/loser/foreign flow lists
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from datasaham import DatasahamAPI
from database import session_scope, get_stock_by_symbol, Stock
from database.models import DailyMover
from config import DATASAHAM_API_KEY, API_RATE_LIMIT


class MarketMoversCollector:
    """Collect daily mover lists and store per-stock flags"""

    def __init__(self, api_key: str = None):
        self.api = DatasahamAPI(api_key or DATASAHAM_API_KEY)

    def _extract_symbol(self, item: Dict[str, Any]) -> Optional[str]:
        symbol = (
            item.get("symbol")
            or item.get("code")
            or item.get("stock_code")
        )
        if not symbol and isinstance(item.get("stock_detail"), dict):
            symbol = item["stock_detail"].get("code") or item["stock_detail"].get("symbol")
        return symbol

    def _extract_score(self, item: Dict[str, Any]) -> Optional[float]:
        # Try common numeric fields in order of relevance
        candidates = [
            ("change", "percentage"),
            ("change", "percent"),
            ("change", "value"),
            ("change", None),
            ("percent", None),
            ("percentage", None),
            ("value", None),
            ("volume", None),
            ("frequency", None),
            ("net", None),
            ("net_foreign", None),
            ("foreign_net", None),
        ]
        for key, subkey in candidates:
            val = item.get(key)
            if subkey and isinstance(val, dict):
                val = val.get(subkey)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.replace(",", "").replace("%", "").strip())
                except ValueError:
                    continue
        return None

    def _fetch_list(self, mover_type: str) -> List[Dict[str, Any]]:
        if mover_type == "top_gainer":
            result = self.api.top_gainer()
        elif mover_type == "top_loser":
            result = self.api.top_loser()
        elif mover_type == "top_value":
            result = self.api.top_value()
        elif mover_type == "top_volume":
            result = self.api.top_volume()
        elif mover_type == "top_frequency":
            result = self.api.top_frequency()
        elif mover_type == "net_foreign_buy":
            result = self.api.net_foreign_buy()
        elif mover_type == "net_foreign_sell":
            result = self.api.net_foreign_sell()
        else:
            return []

        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, dict):
                return data.get("mover_list", data.get("data", [])) or []
            if isinstance(data, list):
                return data
        if isinstance(result, list):
            return result
        return []

    def collect_and_save(
        self,
        date: datetime = None,
        mover_types: List[str] = None
    ) -> Dict[str, int]:
        """Collect and save daily mover lists"""
        stats = {"lists": 0, "records": 0, "unknown_symbols": 0}
        date = date or datetime.now()
        mover_types = mover_types or [
            "top_gainer",
            "top_loser",
            "top_value",
            "top_volume",
            "top_frequency",
            "net_foreign_buy",
            "net_foreign_sell",
        ]

        records: List[Dict[str, Any]] = []

        for mover_type in tqdm(mover_types, desc="Collecting movers"):
            items = self._fetch_list(mover_type)
            stats["lists"] += 1
            for idx, item in enumerate(items, start=1):
                symbol = self._extract_symbol(item)
                if not symbol:
                    continue
                score = self._extract_score(item)
                records.append({
                    "symbol": symbol,
                    "date": date.date(),
                    "mover_type": mover_type,
                    "rank": idx,
                    "score": score,
                })
            time.sleep(API_RATE_LIMIT)

        if not records:
            return stats

        with session_scope() as session:
            for rec in records:
                stock = get_stock_by_symbol(session, rec["symbol"])
                if not stock:
                    stats["unknown_symbols"] += 1
                    continue
                self._upsert_mover(
                    session,
                    stock_id=stock.id,
                    date=rec["date"],
                    mover_type=rec["mover_type"],
                    rank=rec["rank"],
                    score=rec["score"],
                )
                stats["records"] += 1

        return stats

    def backfill_from_db(
        self,
        start_date: str = None,
        end_date: str = None,
        top_n: int = 50
    ) -> Dict[str, int]:
        """
        Backfill movers from existing daily_prices data (no API dependency)
        """
        stats = {"dates": 0, "records": 0}
        from database import session_scope, Stock, DailyPrice

        with session_scope() as session:
            query = session.query(
                DailyPrice.date,
                Stock.symbol,
                DailyPrice.value,
                DailyPrice.volume,
                DailyPrice.frequency,
                DailyPrice.change_percent,
                DailyPrice.foreign_net
            ).join(Stock, Stock.id == DailyPrice.stock_id)

            if start_date:
                query = query.filter(DailyPrice.date >= start_date)
            if end_date:
                query = query.filter(DailyPrice.date <= end_date)

            rows = query.all()

        if not rows:
            return stats

        df = pd.DataFrame(rows, columns=[
            "date", "symbol", "value", "volume", "frequency", "change_percent", "foreign_net"
        ])
        df["date"] = pd.to_datetime(df["date"])

        mover_map = {
            "top_value": ("value", False),
            "top_volume": ("volume", False),
            "top_frequency": ("frequency", False),
            "top_gainer": ("change_percent", False),
            "top_loser": ("change_percent", True),
            "net_foreign_buy": ("foreign_net", False),
            "net_foreign_sell": ("foreign_net", True),
        }

        with session_scope() as session:
            for date, group in df.groupby("date"):
                stats["dates"] += 1
                for mover_type, (col, asc) in mover_map.items():
                    if col not in group.columns:
                        continue
                    series = group[["symbol", col]].dropna()
                    if series.empty:
                        continue
                    ranked = series.sort_values(col, ascending=asc).head(top_n)
                    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                        stock = get_stock_by_symbol(session, row.symbol)
                        if not stock:
                            continue
                        self._upsert_mover(
                            session,
                            stock_id=stock.id,
                            date=date.date(),
                            mover_type=mover_type,
                            rank=rank,
                            score=float(getattr(row, col)),
                        )
                        stats["records"] += 1

        return stats

    def _upsert_mover(self, session, stock_id: int, date, mover_type: str, rank: int, score: Optional[float]) -> None:
        """Upsert a mover record"""
        from sqlalchemy.dialects.sqlite import insert

        record = {
            "stock_id": stock_id,
            "date": date,
            "mover_type": mover_type,
            "rank": rank,
            "score": score,
        }
        stmt = insert(DailyMover).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "date", "mover_type"],
            set_={
                "rank": stmt.excluded.rank,
                "score": stmt.excluded.score,
            }
        )
        session.execute(stmt)


if __name__ == "__main__":
    from database import init_db
    init_db()
    collector = MarketMoversCollector()
    print(collector.collect_and_save())
