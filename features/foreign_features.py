"""
Foreign Flow Features for ML Model
Net foreign buy/sell patterns and accumulation metrics
"""

import pandas as pd
import numpy as np
from typing import List


class ForeignFlowFeatures:
    """Extract foreign flow features - unique to Indonesian market"""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]

    def calculate_net_foreign(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate net foreign buy/sell metrics

        Args:
            df: DataFrame with 'foreign_buy', 'foreign_sell', 'foreign_net' columns

        Returns:
            DataFrame with net foreign features
        """
        result = pd.DataFrame(index=df.index)

        # Use foreign_net if available, otherwise calculate
        if 'foreign_net' in df.columns:
            net = df['foreign_net']
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            net = df['foreign_buy'] - df['foreign_sell']
        else:
            return result

        result['foreign_net'] = net

        # Foreign buy/sell if available
        if 'foreign_buy' in df.columns:
            result['foreign_buy'] = df['foreign_buy']
        if 'foreign_sell' in df.columns:
            result['foreign_sell'] = df['foreign_sell']

        # Net foreign moving averages
        for period in self.lookback_periods:
            result[f'foreign_net_ma_{period}d'] = net.rolling(window=period).mean()

        return result

    def calculate_foreign_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate foreign flow as ratio of trading value

        Args:
            df: DataFrame with foreign and value columns

        Returns:
            DataFrame with foreign ratio features
        """
        result = pd.DataFrame(index=df.index)

        # Get value column
        if 'value' in df.columns:
            value = df['value']
        elif 'volume' in df.columns and 'close' in df.columns:
            value = df['volume'] * df['close']
        else:
            return result

        # Foreign buy ratio
        if 'foreign_buy' in df.columns:
            result['foreign_buy_ratio'] = df['foreign_buy'] / (value + 1)

        # Foreign sell ratio
        if 'foreign_sell' in df.columns:
            result['foreign_sell_ratio'] = df['foreign_sell'] / (value + 1)

        # Net foreign ratio
        if 'foreign_net' in df.columns:
            result['foreign_net_ratio'] = df['foreign_net'] / (value + 1)
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            result['foreign_net_ratio'] = (df['foreign_buy'] - df['foreign_sell']) / (value + 1)

        return result

    def calculate_foreign_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate foreign flow trend indicators

        Args:
            df: DataFrame with foreign flow columns

        Returns:
            DataFrame with foreign trend features
        """
        result = pd.DataFrame(index=df.index)

        # Get net foreign
        if 'foreign_net' in df.columns:
            net = df['foreign_net']
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            net = df['foreign_buy'] - df['foreign_sell']
        else:
            return result

        # Foreign flow change
        for period in [1, 5, 10]:
            result[f'foreign_change_{period}d'] = net.diff(period)

        # Foreign momentum (short MA vs long MA)
        net_ma_5 = net.rolling(window=5).mean()
        net_ma_20 = net.rolling(window=20).mean()
        result['foreign_momentum'] = net_ma_5 - net_ma_20

        # Consecutive days of buying/selling
        is_buy = net > 0
        is_sell = net < 0

        # Count consecutive buy days
        buy_groups = (~is_buy).cumsum()
        result['consecutive_buy_days'] = is_buy.groupby(buy_groups).cumsum()

        # Count consecutive sell days
        sell_groups = (~is_sell).cumsum()
        result['consecutive_sell_days'] = is_sell.groupby(sell_groups).cumsum()

        return result

    def calculate_foreign_accumulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative foreign flow (accumulation/distribution)

        Args:
            df: DataFrame with foreign flow columns

        Returns:
            DataFrame with accumulation features
        """
        result = pd.DataFrame(index=df.index)

        # Get net foreign
        if 'foreign_net' in df.columns:
            net = df['foreign_net']
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            net = df['foreign_buy'] - df['foreign_sell']
        else:
            return result

        # Cumulative foreign flow
        result['foreign_cumulative'] = net.cumsum()

        # Rolling cumulative (sum over period)
        for period in self.lookback_periods:
            result[f'foreign_sum_{period}d'] = net.rolling(window=period).sum()

        # Relative cumulative (vs 50-day sum)
        sum_50d = net.rolling(window=50).sum()
        result['foreign_accumulation_rate'] = net.rolling(window=5).sum() / (sum_50d + 1)

        return result

    def calculate_foreign_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate foreign flow strength indicators

        Args:
            df: DataFrame with foreign flow columns

        Returns:
            DataFrame with strength features
        """
        result = pd.DataFrame(index=df.index)

        # Get net foreign
        if 'foreign_net' in df.columns:
            net = df['foreign_net']
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            net = df['foreign_buy'] - df['foreign_sell']
        else:
            return result

        # Foreign Strength Index (similar to RSI but for foreign flow)
        delta = net.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        result['foreign_strength_index'] = 100 - (100 / (1 + rs))

        # Net foreign Z-score (standardized)
        net_mean = net.rolling(window=20).mean()
        net_std = net.rolling(window=20).std()
        result['foreign_zscore'] = (net - net_mean) / (net_std + 1e-10)

        # Foreign pressure (% of days with net buy in last N days)
        is_buy = net > 0
        for period in [5, 10, 20]:
            result[f'foreign_buy_pct_{period}d'] = is_buy.rolling(window=period).mean()

        return result

    def calculate_foreign_price_relation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relationship between foreign flow and price movement

        Args:
            df: DataFrame with foreign and price columns

        Returns:
            DataFrame with foreign-price relation features
        """
        result = pd.DataFrame(index=df.index)

        # Get net foreign and price change
        if 'foreign_net' in df.columns:
            net = df['foreign_net']
        elif 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            net = df['foreign_buy'] - df['foreign_sell']
        else:
            return result

        if 'close' not in df.columns:
            return result

        price_change = df['close'].pct_change()

        # Foreign flow aligned with price direction?
        # 1 = foreign buying AND price up, or foreign selling AND price down
        # -1 = foreign buying but price down, or foreign selling but price up
        foreign_direction = np.sign(net)
        price_direction = np.sign(price_change)
        result['foreign_price_alignment'] = foreign_direction * price_direction

        # Rolling alignment score
        result['alignment_score_10d'] = result['foreign_price_alignment'].rolling(window=10).mean()

        # Smart money indicator: foreign buys before price increase
        # Lagged correlation between foreign net and future returns
        result['foreign_leads_price'] = net.shift(1) * price_change

        return result

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all foreign flow features

        Args:
            df: DataFrame with OHLCV and foreign flow columns

        Returns:
            DataFrame with all foreign flow features
        """
        features = pd.concat([
            self.calculate_net_foreign(df),
            self.calculate_foreign_ratio(df),
            self.calculate_foreign_trend(df),
            self.calculate_foreign_accumulation(df),
            self.calculate_foreign_strength(df),
            self.calculate_foreign_price_relation(df),
        ], axis=1)

        return features


if __name__ == "__main__":
    # Test with sample data
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    df = pd.DataFrame({
        'close': 1000 + np.cumsum(np.random.randn(100) * 10),
        'volume': np.random.randint(100000, 1000000, 100),
        'value': np.random.randint(1000000, 10000000, 100),
        'foreign_buy': np.random.randint(100000, 5000000, 100),
        'foreign_sell': np.random.randint(100000, 5000000, 100),
    }, index=dates)

    df['foreign_net'] = df['foreign_buy'] - df['foreign_sell']

    # Extract features
    extractor = ForeignFlowFeatures()
    features = extractor.extract_all(df)

    print("Foreign Flow Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data (last 5 rows):")
    print(features.tail())
