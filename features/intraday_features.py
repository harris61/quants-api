"""
Intraday Features for ML Model
Hourly patterns, session analysis, intraday volatility
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class IntradayFeatures:
    """Extract features from intraday (hourly) OHLCV data"""

    def __init__(self):
        # IDX trading hours: 9:00-16:00 (with break 11:30-13:30)
        self.first_hour = 9
        self.last_hour = 15  # Last candle starts at 15:00

    def load_intraday_data(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """Load intraday price data from database"""
        from database import session_scope, get_stock_by_symbol
        from database.models import IntradayPrice

        with session_scope() as session:
            stock = get_stock_by_symbol(session, symbol)
            if not stock:
                return pd.DataFrame()

            query = session.query(IntradayPrice).filter(
                IntradayPrice.stock_id == stock.id
            )
            if start_date:
                query = query.filter(IntradayPrice.date >= start_date)

            records = query.order_by(IntradayPrice.datetime).all()

            if not records:
                return pd.DataFrame()

            data = [{
                'datetime': r.datetime,
                'date': r.date,
                'hour': r.hour,
                'open': r.open or 0,
                'high': r.high or 0,
                'low': r.low or 0,
                'close': r.close or 0,
                'volume': r.volume or 0,
                'value': r.value or 0,
            } for r in records]

            return pd.DataFrame(data)

    def calculate_session_returns(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate session-specific returns
        - First hour return
        - Last hour return
        - Midday return
        """
        if intraday_df.empty:
            return pd.DataFrame()

        results = []

        for date in intraday_df['date'].unique():
            day_data = intraday_df[intraday_df['date'] == date].sort_values('hour')

            if len(day_data) < 2:
                continue

            # First hour (9:00-10:00)
            first_hour_data = day_data[day_data['hour'] == self.first_hour]
            if not first_hour_data.empty:
                first_open = first_hour_data['open'].iloc[0]
                first_close = first_hour_data['close'].iloc[0]
                first_hour_return = (first_close - first_open) / first_open if first_open else 0
            else:
                first_hour_return = 0
                first_open = day_data['open'].iloc[0]

            # Last hour (15:00-16:00)
            last_hour_data = day_data[day_data['hour'] == self.last_hour]
            if not last_hour_data.empty:
                last_open = last_hour_data['open'].iloc[0]
                last_close = last_hour_data['close'].iloc[0]
                last_hour_return = (last_close - last_open) / last_open if last_open else 0
            else:
                last_hour_return = 0
                last_close = day_data['close'].iloc[-1]

            # Full day return from intraday data
            day_return = (last_close - first_open) / first_open if first_open else 0

            # First vs last hour comparison
            first_last_ratio = first_hour_return / (abs(last_hour_return) + 1e-10)

            results.append({
                'date': date,
                'intraday_first_hour_return': first_hour_return,
                'intraday_last_hour_return': last_hour_return,
                'intraday_day_return': day_return,
                'intraday_first_last_ratio': first_last_ratio,
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_intraday_volatility(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday volatility metrics
        - Hourly return std
        - High-low range
        - True range
        """
        if intraday_df.empty:
            return pd.DataFrame()

        results = []

        for date in intraday_df['date'].unique():
            day_data = intraday_df[intraday_df['date'] == date].sort_values('hour')

            if len(day_data) < 2:
                continue

            # Calculate hourly returns
            hourly_returns = day_data['close'].pct_change().dropna()

            # Intraday volatility (std of hourly returns)
            intraday_vol = hourly_returns.std() if len(hourly_returns) > 1 else 0

            # Intraday range
            day_high = day_data['high'].max()
            day_low = day_data['low'].min()
            day_open = day_data['open'].iloc[0]
            intraday_range = (day_high - day_low) / day_open if day_open else 0

            # Sum of hourly ranges (total price movement)
            hourly_ranges = (day_data['high'] - day_data['low']) / day_data['open'].replace(0, 1)
            total_hourly_range = hourly_ranges.sum()

            results.append({
                'date': date,
                'intraday_volatility': intraday_vol,
                'intraday_range_pct': intraday_range,
                'intraday_total_hourly_range': total_hourly_range,
                'intraday_avg_hourly_range': hourly_ranges.mean(),
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_volume_distribution(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday volume distribution
        - First hour volume ratio
        - Last hour volume ratio
        - Volume concentration
        """
        if intraday_df.empty:
            return pd.DataFrame()

        results = []

        for date in intraday_df['date'].unique():
            day_data = intraday_df[intraday_df['date'] == date].sort_values('hour')

            if len(day_data) < 2:
                continue

            total_volume = day_data['volume'].sum()
            if total_volume == 0:
                continue

            # First hour volume
            first_hour_data = day_data[day_data['hour'] == self.first_hour]
            first_hour_vol = first_hour_data['volume'].sum() if not first_hour_data.empty else 0
            first_hour_vol_ratio = first_hour_vol / total_volume

            # Last hour volume
            last_hour_data = day_data[day_data['hour'] == self.last_hour]
            last_hour_vol = last_hour_data['volume'].sum() if not last_hour_data.empty else 0
            last_hour_vol_ratio = last_hour_vol / total_volume

            # Volume HHI (concentration)
            vol_shares = (day_data['volume'] / total_volume) ** 2
            volume_hhi = vol_shares.sum()

            results.append({
                'date': date,
                'intraday_first_hour_vol_ratio': first_hour_vol_ratio,
                'intraday_last_hour_vol_ratio': last_hour_vol_ratio,
                'intraday_volume_hhi': volume_hhi,
                'intraday_first_last_vol_ratio': first_hour_vol_ratio / (last_hour_vol_ratio + 1e-10),
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_price_patterns(self, intraday_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday price patterns
        - Gap and go (first hour vs gap)
        - Reversal indicators
        """
        if intraday_df.empty:
            return pd.DataFrame()

        results = []

        for date in intraday_df['date'].unique():
            day_data = intraday_df[intraday_df['date'] == date].sort_values('hour')

            if len(day_data) < 2:
                continue

            # Check if first hour continues or reverses the gap
            first_open = day_data['open'].iloc[0]
            first_hour_data = day_data[day_data['hour'] == self.first_hour]
            first_close = first_hour_data['close'].iloc[0] if not first_hour_data.empty else first_open
            last_close = day_data['close'].iloc[-1]

            # Day high and low
            day_high = day_data['high'].max()
            day_low = day_data['low'].min()

            # Close position in day range (0 = closed at low, 1 = closed at high)
            if day_high != day_low:
                close_position = (last_close - day_low) / (day_high - day_low)
            else:
                close_position = 0.5

            # Morning continuation (did first hour direction continue?)
            first_direction = np.sign(first_close - first_open)
            day_direction = np.sign(last_close - first_open)
            morning_continuation = int(first_direction == day_direction)

            results.append({
                'date': date,
                'intraday_close_position': close_position,
                'intraday_morning_continuation': morning_continuation,
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_rolling_intraday_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages of intraday features"""
        if features_df.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=features_df.index)

        # Rolling averages for key metrics
        for period in [5, 10]:
            if 'intraday_volatility' in features_df.columns:
                result[f'intraday_vol_ma_{period}d'] = features_df['intraday_volatility'].rolling(period, min_periods=1).mean()

            if 'intraday_first_hour_return' in features_df.columns:
                result[f'intraday_first_hour_return_ma_{period}d'] = features_df['intraday_first_hour_return'].rolling(period, min_periods=1).mean()

            if 'intraday_morning_continuation' in features_df.columns:
                result[f'intraday_continuation_rate_{period}d'] = features_df['intraday_morning_continuation'].rolling(period, min_periods=1).mean()

        return result

    def extract_all(self, df: pd.DataFrame, symbol: str = None, intraday_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all intraday features

        Args:
            df: Daily price DataFrame (used for index alignment)
            symbol: Stock symbol (to load intraday data)
            intraday_df: Pre-loaded intraday data (optional)

        Returns:
            DataFrame with all intraday features aligned to daily dates
        """
        if intraday_df is None and symbol:
            intraday_df = self.load_intraday_data(symbol)

        if intraday_df is None or intraday_df.empty:
            return pd.DataFrame(index=df.index)

        # Calculate all feature groups
        session_returns = self.calculate_session_returns(intraday_df)
        volatility = self.calculate_intraday_volatility(intraday_df)
        volume_dist = self.calculate_volume_distribution(intraday_df)
        patterns = self.calculate_price_patterns(intraday_df)

        # Combine features
        feature_dfs = [session_returns, volatility, volume_dist, patterns]
        features = pd.concat([f for f in feature_dfs if not f.empty], axis=1)

        # Add rolling features
        rolling_features = self.calculate_rolling_intraday_features(features)
        if not rolling_features.empty:
            features = pd.concat([features, rolling_features], axis=1)

        # Align to daily price DataFrame index
        features = features.reindex(df.index)

        return features


if __name__ == "__main__":
    # Test with sample data
    from datetime import datetime, timedelta

    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    np.random.seed(42)

    # Create sample intraday data
    intraday_data = []
    for date in dates:
        for hour in [9, 10, 11, 13, 14, 15]:  # IDX trading hours
            base_price = 1000 + np.random.randn() * 10
            intraday_data.append({
                'datetime': datetime.combine(date, datetime.min.time().replace(hour=hour)),
                'date': date,
                'hour': hour,
                'open': base_price,
                'high': base_price + np.random.uniform(0, 10),
                'low': base_price - np.random.uniform(0, 10),
                'close': base_price + np.random.randn() * 5,
                'volume': np.random.randint(10000, 100000),
                'value': np.random.randint(10000000, 100000000),
            })

    intraday_df = pd.DataFrame(intraday_data)

    # Create sample daily price data
    price_df = pd.DataFrame({
        'close': 1000 + np.cumsum(np.random.randn(10) * 10),
        'volume': np.random.randint(100000, 1000000, 10),
    }, index=dates)

    # Extract features
    extractor = IntradayFeatures()
    features = extractor.extract_all(price_df, intraday_df=intraday_df)

    print("Intraday Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data:")
    print(features.head())
