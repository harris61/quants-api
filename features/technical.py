"""
Technical Indicators for ML Model
RSI, MACD, Moving Averages, Bollinger Bands, etc.
"""

import pandas as pd
import numpy as np
from typing import List

from config import MA_PERIODS


class TechnicalFeatures:
    """Extract technical analysis features"""

    def __init__(self, ma_periods: List[int] = None):
        self.ma_periods = ma_periods or MA_PERIODS

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with MA features
        """
        result = pd.DataFrame(index=df.index)
        close = df['close']

        for period in self.ma_periods:
            # Simple Moving Average
            result[f'sma_{period}'] = close.rolling(window=period).mean()

            # Exponential Moving Average
            result[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()

            # Price relative to MA
            result[f'close_to_sma_{period}'] = close / result[f'sma_{period}']

        # MA crossovers
        if 5 in self.ma_periods and 20 in self.ma_periods:
            result['sma_5_above_20'] = (result['sma_5'] > result['sma_20']).astype(int)

        if 10 in self.ma_periods and 50 in self.ma_periods:
            result['sma_10_above_50'] = (result['sma_10'] > result['sma_50']).astype(int)

        return result

    def calculate_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Relative Strength Index using Wilder's smoothing method

        Args:
            df: DataFrame with 'close' column
            periods: RSI periods (default [7, 14, 21])

        Returns:
            DataFrame with RSI features
        """
        result = pd.DataFrame(index=df.index)
        periods = periods or [7, 14, 21]

        delta = df['close'].diff()

        for period in periods:
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            # Use Wilder's smoothing (EMA with alpha = 1/period)
            # This is the standard RSI calculation
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            result[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # RSI zones
        if 'rsi_14' in result.columns:
            result['rsi_oversold'] = (result['rsi_14'] < 30).astype(int)
            result['rsi_overbought'] = (result['rsi_14'] > 70).astype(int)

        return result

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with MACD features
        """
        result = pd.DataFrame(index=df.index)
        close = df['close']

        # Standard MACD settings: 12, 26, 9
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        result['macd_line'] = ema_12 - ema_26
        result['macd_signal'] = result['macd_line'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']

        # MACD crossover signals
        result['macd_bullish'] = (
            (result['macd_line'] > result['macd_signal']) &
            (result['macd_line'].shift(1) <= result['macd_signal'].shift(1))
        ).astype(int)

        result['macd_bearish'] = (
            (result['macd_line'] < result['macd_signal']) &
            (result['macd_line'].shift(1) >= result['macd_signal'].shift(1))
        ).astype(int)

        # MACD above/below zero
        result['macd_above_zero'] = (result['macd_line'] > 0).astype(int)

        return result

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with 'close' column
            period: MA period for middle band
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band features
        """
        result = pd.DataFrame(index=df.index)
        close = df['close']

        # Middle band (SMA)
        result['bb_middle'] = close.rolling(window=period).mean()

        # Standard deviation
        rolling_std = close.rolling(window=period).std()

        # Upper and lower bands
        result['bb_upper'] = result['bb_middle'] + (rolling_std * std_dev)
        result['bb_lower'] = result['bb_middle'] - (rolling_std * std_dev)

        # Bandwidth
        result['bb_bandwidth'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']

        # %B indicator (position within bands)
        result['bb_percent_b'] = (close - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)

        # Touch bands
        result['bb_touch_upper'] = (close >= result['bb_upper']).astype(int)
        result['bb_touch_lower'] = (close <= result['bb_lower']).astype(int)

        return result

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period
            d_period: %D period (signal line)

        Returns:
            DataFrame with Stochastic features
        """
        result = pd.DataFrame(index=df.index)

        # Lowest low and highest high
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()

        # %K
        result['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100

        # %D (signal line)
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()

        # Stochastic zones
        result['stoch_oversold'] = (result['stoch_k'] < 20).astype(int)
        result['stoch_overbought'] = (result['stoch_k'] > 80).astype(int)

        return result

    def calculate_atr(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Average True Range

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            periods: ATR periods

        Returns:
            DataFrame with ATR features
        """
        result = pd.DataFrame(index=df.index)
        periods = periods or [7, 14, 21]

        # True Range
        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - prev_close),
                abs(df['low'] - prev_close)
            )
        )

        for period in periods:
            result[f'atr_{period}'] = tr.rolling(window=period).mean()
            # ATR as percentage of close
            result[f'atr_{period}_pct'] = result[f'atr_{period}'] / df['close'] * 100

        return result

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) using Wilder's smoothing

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ADX period

        Returns:
            DataFrame with ADX features
        """
        result = pd.DataFrame(index=df.index)

        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()  # Note: low_diff is inverted

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # True Range
        prev_close = df['close'].shift(1)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - prev_close),
                abs(df['low'] - prev_close)
            )
        )

        # Use Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1 / period
        tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # +DI and -DI
        result['plus_di'] = (plus_dm_smooth / (tr_smooth + 1e-10)) * 100
        result['minus_di'] = (minus_dm_smooth / (tr_smooth + 1e-10)) * 100

        # DX and ADX (ADX is smoothed DX)
        dx = (abs(result['plus_di'] - result['minus_di']) /
              (result['plus_di'] + result['minus_di'] + 1e-10)) * 100
        result['adx'] = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # Trend strength
        result['strong_trend'] = (result['adx'] > 25).astype(int)

        return result

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: CCI period

        Returns:
            DataFrame with CCI features
        """
        result = pd.DataFrame(index=df.index)

        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # SMA of typical price
        tp_sma = tp.rolling(window=period).mean()

        # Mean deviation
        mean_dev = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        # CCI
        result['cci'] = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)

        # CCI zones
        result['cci_oversold'] = (result['cci'] < -100).astype(int)
        result['cci_overbought'] = (result['cci'] > 100).astype(int)

        return result

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Williams %R

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Williams %R period

        Returns:
            DataFrame with Williams %R features
        """
        result = pd.DataFrame(index=df.index)

        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()

        result['williams_r'] = ((highest_high - df['close']) /
                                (highest_high - lowest_low + 1e-10)) * -100

        return result

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all technical features

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all technical features
        """
        features = pd.concat([
            self.calculate_moving_averages(df),
            self.calculate_rsi(df),
            self.calculate_macd(df),
            self.calculate_bollinger_bands(df),
            self.calculate_stochastic(df),
            self.calculate_atr(df),
            self.calculate_adx(df),
            self.calculate_cci(df),
            self.calculate_williams_r(df),
        ], axis=1)

        return features


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    close = 1000 + np.cumsum(np.random.randn(100) * 10)
    df = pd.DataFrame({
        'open': close + np.random.randn(100) * 5,
        'close': close,
        'volume': np.random.randint(100000, 1000000, 100),
    }, index=dates)

    # Generate high/low based on open/close
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(100) * 10
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(100) * 10

    # Extract features
    extractor = TechnicalFeatures()
    features = extractor.extract_all(df)

    print("Technical Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data (last 5 rows):")
    print(features[['sma_20', 'rsi_14', 'macd_line', 'bb_percent_b', 'adx']].tail())
