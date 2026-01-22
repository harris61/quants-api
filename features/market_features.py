"""
Market regime features placeholder.
"""

import pandas as pd


class MarketRegimeFeatures:
    """Minimal market regime features (placeholder)."""

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        return pd.DataFrame(index=df.index)
