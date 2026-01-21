"""
Volume-based Features for ML Model
Volume patterns, liquidity, and trading activity metrics
"""

import pandas as pd
import numpy as np
from typing import List


class VolumeFeatures:
    """Extract volume-based features from trading data"""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]

    def calculate_volume_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume ratios compared to moving averages

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume ratio features
        """
        result = pd.DataFrame(index=df.index)

        for period in self.lookback_periods:
            vol_ma = df['volume'].rolling(window=period).mean()
            result[f'volume_ratio_{period}d'] = df['volume'] / vol_ma

        # Volume ratio vs previous day
        result['volume_ratio_1d'] = df['volume'] / df['volume'].shift(1)

        return result

    def calculate_volume_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume trend indicators

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume trend features
        """
        result = pd.DataFrame(index=df.index)

        # Volume moving averages
        for period in self.lookback_periods:
            result[f'volume_ma_{period}d'] = df['volume'].rolling(window=period).mean()

        # Volume change
        for period in [1, 5, 10]:
            result[f'volume_change_{period}d'] = df['volume'].pct_change(period)

        # Volume momentum (short MA vs long MA)
        if 'volume_ma_5d' in result.columns and 'volume_ma_20d' in result.columns:
            result['volume_momentum'] = result['volume_ma_5d'] / result['volume_ma_20d']

        # Days since volume spike (volume > 2x average)
        vol_ma_20 = df['volume'].rolling(window=20).mean()
        volume_spike = df['volume'] > (vol_ma_20 * 2)
        result['volume_spike'] = volume_spike.astype(int)

        return result

    def calculate_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading value (volume * price) features

        Args:
            df: DataFrame with 'volume', 'close', 'value' columns

        Returns:
            DataFrame with value features
        """
        result = pd.DataFrame(index=df.index)

        # Calculate value if not present
        if 'value' in df.columns:
            value = df['value']
        else:
            value = df['volume'] * df['close']

        result['value'] = value

        # Value moving averages
        for period in self.lookback_periods:
            result[f'value_ma_{period}d'] = value.rolling(window=period).mean()

        # Value ratio
        result['value_ratio_20d'] = value / result['value_ma_20d']

        # Average trade size (value / frequency if available)
        if 'frequency' in df.columns:
            result['avg_trade_size'] = value / (df['frequency'] + 1)

        return result

    def calculate_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transaction frequency features

        Args:
            df: DataFrame with 'frequency' column

        Returns:
            DataFrame with frequency features
        """
        result = pd.DataFrame(index=df.index)

        if 'frequency' not in df.columns:
            return result

        # Frequency moving averages
        for period in [5, 10, 20]:
            result[f'frequency_ma_{period}d'] = df['frequency'].rolling(window=period).mean()

        # Frequency ratio
        freq_ma_20 = df['frequency'].rolling(window=20).mean()
        result['frequency_ratio_20d'] = df['frequency'] / freq_ma_20

        # Frequency change
        result['frequency_change_1d'] = df['frequency'].pct_change(1)
        result['frequency_change_5d'] = df['frequency'].pct_change(5)

        return result

    def calculate_volume_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate combined volume-price features

        Args:
            df: DataFrame with 'volume', 'close' columns

        Returns:
            DataFrame with volume-price features
        """
        result = pd.DataFrame(index=df.index)

        # Price change
        price_change = df['close'].pct_change()

        # Volume-weighted price change (higher volume = more significant)
        vol_ma = df['volume'].rolling(window=20).mean()
        vol_weight = df['volume'] / vol_ma
        result['vwpc'] = price_change * vol_weight

        # On-Balance Volume (OBV)
        # Fix: fill first NaN with 0 to prevent NaN propagation
        obv_direction = np.sign(price_change.fillna(0))
        obv = (obv_direction * df['volume']).cumsum()
        result['obv'] = obv
        result['obv_ma_20'] = obv.rolling(window=20).mean()
        result['obv_ratio'] = obv / (result['obv_ma_20'] + 1e-10)

        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        ad = (clv * df['volume']).cumsum()
        result['ad_line'] = ad

        # Volume-Price Trend (VPT)
        # Fix: fill first NaN with 0
        vpt = (price_change.fillna(0) * df['volume']).cumsum()
        result['vpt'] = vpt

        # Chaikin Money Flow (20-day)
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        mf_volume = mf_multiplier * df['volume']
        result['cmf_20'] = mf_volume.rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)

        return result

    def calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity-related features

        Args:
            df: DataFrame with 'volume', 'high', 'low', 'close' columns

        Returns:
            DataFrame with liquidity features
        """
        result = pd.DataFrame(index=df.index)

        # Amihud illiquidity ratio (abs return / volume)
        abs_return = abs(df['close'].pct_change())
        if 'value' in df.columns:
            result['amihud_illiq'] = abs_return / (df['value'] + 1)
        else:
            result['amihud_illiq'] = abs_return / (df['volume'] * df['close'] + 1)

        # Rolling average illiquidity
        result['amihud_illiq_20d'] = result['amihud_illiq'].rolling(window=20).mean()

        # Turnover (volume / market cap proxy - assuming outstanding shares)
        # Without market cap, we use relative volume
        result['relative_volume'] = df['volume'] / df['volume'].rolling(window=50).mean()

        return result

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all volume-based features

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all volume features
        """
        features = pd.concat([
            self.calculate_volume_ratios(df),
            self.calculate_volume_trend(df),
            self.calculate_value_features(df),
            self.calculate_frequency_features(df),
            self.calculate_volume_price_features(df),
            self.calculate_liquidity_features(df),
        ], axis=1)

        return features


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'open': 1000 + np.cumsum(np.random.randn(100) * 10),
        'high': 1000 + np.cumsum(np.random.randn(100) * 10) + np.random.rand(100) * 20,
        'low': 1000 + np.cumsum(np.random.randn(100) * 10) - np.random.rand(100) * 20,
        'close': 1000 + np.cumsum(np.random.randn(100) * 10),
        'volume': np.random.randint(100000, 1000000, 100),
        'value': np.random.randint(1000000, 10000000, 100),
        'frequency': np.random.randint(100, 1000, 100),
    }, index=dates)

    # Fix high/low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    # Extract features
    extractor = VolumeFeatures()
    features = extractor.extract_all(df)

    print("Volume Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data (last 5 rows):")
    print(features.tail())
