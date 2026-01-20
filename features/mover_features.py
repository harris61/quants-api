"""
Market Movers Features for ML Model
Top value/volume/frequency/gainer/loser/foreign flow lists
"""

import pandas as pd
from typing import List, Optional


class MoverFeatures:
    """Extract features from daily mover lists"""

    def __init__(self, mover_types: Optional[List[str]] = None):
        self.mover_types = mover_types or [
            "top_gainer",
            "top_loser",
            "top_value",
            "top_volume",
            "top_frequency",
            "net_foreign_buy",
            "net_foreign_sell",
        ]

    def load_mover_data(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """Load mover data from database"""
        from database import session_scope, get_stock_by_symbol
        from database.models import DailyMover

        with session_scope() as session:
            stock = get_stock_by_symbol(session, symbol)
            if not stock:
                return pd.DataFrame()

            query = session.query(DailyMover).filter(
                DailyMover.stock_id == stock.id
            )
            if start_date:
                query = query.filter(DailyMover.date >= start_date)

            records = query.order_by(DailyMover.date).all()
            if not records:
                return pd.DataFrame()

            data = [{
                "date": r.date,
                "mover_type": r.mover_type,
                "rank": r.rank,
                "score": r.score,
            } for r in records]

            return pd.DataFrame(data)

    def extract_all(self, df: pd.DataFrame, symbol: str = None, mover_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all mover features

        Args:
            df: Price DataFrame (used for index alignment)
            symbol: Stock symbol (to load mover data)
            mover_df: Pre-loaded mover data (optional)

        Returns:
            DataFrame with mover features aligned to price dates
        """
        if mover_df is None and symbol:
            mover_df = self.load_mover_data(symbol)

        if mover_df is None or mover_df.empty:
            return pd.DataFrame(index=df.index)

        mover_df = mover_df.copy()
        mover_df["date"] = pd.to_datetime(mover_df["date"])
        mover_df = mover_df[mover_df["mover_type"].isin(self.mover_types)]

        # Build feature frame
        features = pd.DataFrame(index=df.index)

        for mover_type in self.mover_types:
            subset = mover_df[mover_df["mover_type"] == mover_type]
            if subset.empty:
                features[f"mover_{mover_type}_flag"] = 0
                features[f"mover_{mover_type}_rank"] = 0
                features[f"mover_{mover_type}_score"] = 0
                continue

            subset = subset.set_index("date")
            rank = subset["rank"].reindex(features.index)
            score = subset["score"].reindex(features.index)

            features[f"mover_{mover_type}_flag"] = rank.notna().astype(int)
            features[f"mover_{mover_type}_rank"] = rank.apply(lambda x: 1 / x if x and x > 0 else 0)
            features[f"mover_{mover_type}_score"] = score.fillna(0)

        return features


if __name__ == "__main__":
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"close": np.arange(5)}, index=dates)
    mover_df = pd.DataFrame({
        "date": dates[:2],
        "mover_type": ["top_value", "top_value"],
        "rank": [1, 3],
        "score": [1000, 500],
    })
    extractor = MoverFeatures()
    print(extractor.extract_all(df, mover_df=mover_df))
