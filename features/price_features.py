"""
Price-based Features for ML Model
Returns, momentum, volatility, and price patterns
"""

import pandas as pd
import numpy as np
from typing import List

from config import RETURN_PERIODS, VOLATILITY_PERIOD, IDX_TRADING_DAYS_PER_YEAR


class PriceFeatures:
    """Extract price-based features from OHLCV data"""

    def __init__(
        self,
        return_periods: List[int] = None,
        volatility_period: int = None
    ):
        self.return_periods = return_periods or RETURN_PERIODS
        self.volatility_period = volatility_period or VOLATILITY_PERIOD

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for multiple periods

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with return columns
        """
        result = pd.DataFrame(index=df.index)

        for period in self.return_periods:
            result[f'return_{period}d'] = df['close'].pct_change(period)

        return result

    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns for multiple periods"""
        result = pd.DataFrame(index=df.index)

        for period in self.return_periods:
            result[f'log_return_{period}d'] = np.log(
                df['close'] / df['close'].shift(period)
            )

        return result

    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility (std of returns)

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with volatility columns
        """
        result = pd.DataFrame(index=df.index)

        # Daily returns
        daily_returns = df['close'].pct_change()

        # Rolling standard deviation
        result[f'volatility_{self.volatility_period}d'] = daily_returns.rolling(
            window=self.volatility_period
        ).std()

        # Annualized volatility (IDX has ~242 trading days per year)
        result['volatility_annualized'] = result[f'volatility_{self.volatility_period}d'] * np.sqrt(IDX_TRADING_DAYS_PER_YEAR)

        return result

    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with momentum features
        """
        result = pd.DataFrame(index=df.index)

        # Rate of change (ROC)
        for period in [5, 10, 20]:
            result[f'roc_{period}d'] = (
                (df['close'] - df['close'].shift(period)) /
                df['close'].shift(period)
            ) * 100

        # Momentum (price diff)
        for period in [5, 10, 20]:
            result[f'momentum_{period}d'] = df['close'] - df['close'].shift(period)

        return result

    def calculate_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price position relative to historical highs/lows

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            DataFrame with price position features
        """
        result = pd.DataFrame(index=df.index)

        # 52-week (252 trading days) high/low position
        high_52w = df['high'].rolling(window=252, min_periods=20).max()
        low_52w = df['low'].rolling(window=252, min_periods=20).min()

        result['pct_from_52w_high'] = (df['close'] - high_52w) / high_52w
        result['pct_from_52w_low'] = (df['close'] - low_52w) / low_52w

        # 20-day high/low position
        high_20d = df['high'].rolling(window=20).max()
        low_20d = df['low'].rolling(window=20).min()

        result['pct_from_20d_high'] = (df['close'] - high_20d) / high_20d
        result['pct_from_20d_low'] = (df['close'] - low_20d) / low_20d

        # Price position in range (0-1)
        result['price_position_20d'] = (df['close'] - low_20d) / (high_20d - low_20d + 1e-10)
        result['price_position_52w'] = (df['close'] - low_52w) / (high_52w - low_52w + 1e-10)

        return result

    def calculate_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap features (open vs previous close)

        Args:
            df: DataFrame with 'open', 'close' columns

        Returns:
            DataFrame with gap features
        """
        result = pd.DataFrame(index=df.index)

        prev_close = df['close'].shift(1)

        # Gap = (Open - Prev Close) / Prev Close
        result['gap'] = (df['open'] - prev_close) / prev_close
        result['gap_pct'] = result['gap'] * 100

        # Gap type (gap up / gap down / no gap)
        result['gap_up'] = (result['gap'] > 0.01).astype(int)
        result['gap_down'] = (result['gap'] < -0.01).astype(int)

        return result

    def calculate_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday price features

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with intraday features
        """
        result = pd.DataFrame(index=df.index)

        # True range
        prev_close = df['close'].shift(1)
        result['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - prev_close),
                abs(df['low'] - prev_close)
            )
        )

        # ATR (Average True Range)
        result['atr_14'] = result['true_range'].rolling(window=14).mean()
        result['atr_pct'] = result['atr_14'] / df['close'] * 100

        # Candle body
        result['body'] = abs(df['close'] - df['open'])
        result['body_pct'] = result['body'] / df['open'] * 100

        # Upper/lower wick
        result['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        result['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']

        # Candle type (bullish/bearish)
        result['bullish_candle'] = (df['close'] > df['open']).astype(int)

        # Intraday range
        result['intraday_range'] = (df['high'] - df['low']) / df['low'] * 100

        return result

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all price-based features

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all price features
        """
        features = pd.concat([
            self.calculate_returns(df),
            self.calculate_volatility(df),
            self.calculate_momentum(df),
            self.calculate_price_position(df),
            self.calculate_gap(df),
            self.calculate_intraday_features(df),
        ], axis=1)

        return features


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 1000 + np.cumsum(np.random.randn(100) * 10),
        'high': 1000 + np.cumsum(np.random.randn(100) * 10) + np.random.rand(100) * 20,
        'low': 1000 + np.cumsum(np.random.randn(100) * 10) - np.random.rand(100) * 20,
        'close': 1000 + np.cumsum(np.random.randn(100) * 10),
        'volume': np.random.randint(100000, 1000000, 100),
    }, index=dates)

    # Fix high/low to be consistent
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    # Extract features
    extractor = PriceFeatures()
    features = extractor.extract_all(df)

    print("Price Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data (last 5 rows):")
    print(features.tail())
