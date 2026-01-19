"""
Insider Trading Features for ML Model
Insider sentiment, transaction patterns, ownership changes
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import timedelta


class InsiderFeatures:
    """Extract features from insider trading data"""

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days

    def load_insider_data(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """Load insider trading data from database"""
        from database import session_scope, get_stock_by_symbol
        from database.models import InsiderTrade

        with session_scope() as session:
            stock = get_stock_by_symbol(session, symbol)
            if not stock:
                return pd.DataFrame()

            query = session.query(InsiderTrade).filter(
                InsiderTrade.stock_id == stock.id
            )
            if start_date:
                query = query.filter(InsiderTrade.transaction_date >= start_date)

            records = query.order_by(InsiderTrade.transaction_date).all()

            if not records:
                return pd.DataFrame()

            data = [{
                'date': r.transaction_date,
                'insider_name': r.insider_name,
                'position': r.position,
                'transaction_type': r.transaction_type,
                'shares': r.shares or 0,
                'price': r.price or 0,
                'value': r.value or 0,
            } for r in records]

            return pd.DataFrame(data)

    def calculate_insider_sentiment(self, insider_df: pd.DataFrame, ref_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate insider sentiment indicators
        - Buy/sell ratio
        - Net insider activity
        """
        result = pd.DataFrame(index=ref_dates)

        if insider_df.empty:
            result['insider_buy_count'] = 0
            result['insider_sell_count'] = 0
            result['insider_net_shares'] = 0
            result['insider_net_value'] = 0
            result['insider_buy_sell_ratio'] = 0
            return result

        # Ensure date column is datetime
        insider_df = insider_df.copy()
        insider_df['date'] = pd.to_datetime(insider_df['date'])

        for ref_date in ref_dates:
            ref_date_ts = pd.Timestamp(ref_date)
            # Look back N days from reference date
            lookback_start = ref_date_ts - timedelta(days=self.lookback_days)
            mask = (insider_df['date'] >= lookback_start) & (insider_df['date'] <= ref_date_ts)
            period_data = insider_df[mask]

            if period_data.empty:
                result.loc[ref_date, 'insider_buy_count'] = 0
                result.loc[ref_date, 'insider_sell_count'] = 0
                result.loc[ref_date, 'insider_net_shares'] = 0
                result.loc[ref_date, 'insider_net_value'] = 0
                result.loc[ref_date, 'insider_buy_sell_ratio'] = 0
                continue

            buys = period_data[period_data['transaction_type'] == 'buy']
            sells = period_data[period_data['transaction_type'] == 'sell']

            buy_count = len(buys)
            sell_count = len(sells)
            buy_shares = buys['shares'].sum()
            sell_shares = sells['shares'].sum()
            buy_value = buys['value'].sum()
            sell_value = sells['value'].sum()

            result.loc[ref_date, 'insider_buy_count'] = buy_count
            result.loc[ref_date, 'insider_sell_count'] = sell_count
            result.loc[ref_date, 'insider_net_shares'] = buy_shares - sell_shares
            result.loc[ref_date, 'insider_net_value'] = buy_value - sell_value
            result.loc[ref_date, 'insider_buy_sell_ratio'] = buy_count / (sell_count + 1)

        return result

    def calculate_insider_recency(self, insider_df: pd.DataFrame, ref_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate recency of insider transactions
        - Days since last buy/sell
        - Recent transaction intensity
        """
        result = pd.DataFrame(index=ref_dates)

        if insider_df.empty:
            result['insider_days_since_last'] = 999
            result['insider_days_since_last_buy'] = 999
            result['insider_days_since_last_sell'] = 999
            return result

        insider_df = insider_df.copy()
        insider_df['date'] = pd.to_datetime(insider_df['date'])

        for ref_date in ref_dates:
            ref_date_ts = pd.Timestamp(ref_date)
            past_trades = insider_df[insider_df['date'] <= ref_date_ts]

            if past_trades.empty:
                result.loc[ref_date, 'insider_days_since_last'] = 999
                result.loc[ref_date, 'insider_days_since_last_buy'] = 999
                result.loc[ref_date, 'insider_days_since_last_sell'] = 999
                continue

            # Days since last trade
            last_trade_date = past_trades['date'].max()
            days_since = (ref_date_ts - last_trade_date).days
            result.loc[ref_date, 'insider_days_since_last'] = days_since

            # Days since last buy
            buys = past_trades[past_trades['transaction_type'] == 'buy']
            if not buys.empty:
                last_buy_date = buys['date'].max()
                result.loc[ref_date, 'insider_days_since_last_buy'] = (ref_date_ts - last_buy_date).days
            else:
                result.loc[ref_date, 'insider_days_since_last_buy'] = 999

            # Days since last sell
            sells = past_trades[past_trades['transaction_type'] == 'sell']
            if not sells.empty:
                last_sell_date = sells['date'].max()
                result.loc[ref_date, 'insider_days_since_last_sell'] = (ref_date_ts - last_sell_date).days
            else:
                result.loc[ref_date, 'insider_days_since_last_sell'] = 999

        return result

    def calculate_insider_magnitude(self, insider_df: pd.DataFrame, ref_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate magnitude of insider transactions
        - Average transaction size
        - Total insider activity value
        """
        result = pd.DataFrame(index=ref_dates)

        if insider_df.empty:
            result['insider_avg_transaction_value'] = 0
            result['insider_total_value'] = 0
            result['insider_unique_insiders'] = 0
            return result

        insider_df = insider_df.copy()
        insider_df['date'] = pd.to_datetime(insider_df['date'])

        for ref_date in ref_dates:
            ref_date_ts = pd.Timestamp(ref_date)
            lookback_start = ref_date_ts - timedelta(days=self.lookback_days)
            mask = (insider_df['date'] >= lookback_start) & (insider_df['date'] <= ref_date_ts)
            period_data = insider_df[mask]

            if period_data.empty:
                result.loc[ref_date, 'insider_avg_transaction_value'] = 0
                result.loc[ref_date, 'insider_total_value'] = 0
                result.loc[ref_date, 'insider_unique_insiders'] = 0
                continue

            result.loc[ref_date, 'insider_avg_transaction_value'] = period_data['value'].mean()
            result.loc[ref_date, 'insider_total_value'] = period_data['value'].sum()
            result.loc[ref_date, 'insider_unique_insiders'] = period_data['insider_name'].nunique()

        return result

    def calculate_insider_position_features(self, insider_df: pd.DataFrame, ref_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate features based on insider position/role
        - Executive vs board transactions
        """
        result = pd.DataFrame(index=ref_dates)

        if insider_df.empty:
            result['insider_exec_buy_count'] = 0
            result['insider_exec_sell_count'] = 0
            return result

        insider_df = insider_df.copy()
        insider_df['date'] = pd.to_datetime(insider_df['date'])

        # Identify executive positions
        executive_keywords = ['director', 'ceo', 'cfo', 'president', 'chief', 'direktur', 'presdir']

        for ref_date in ref_dates:
            ref_date_ts = pd.Timestamp(ref_date)
            lookback_start = ref_date_ts - timedelta(days=self.lookback_days)
            mask = (insider_df['date'] >= lookback_start) & (insider_df['date'] <= ref_date_ts)
            period_data = insider_df[mask]

            if period_data.empty:
                result.loc[ref_date, 'insider_exec_buy_count'] = 0
                result.loc[ref_date, 'insider_exec_sell_count'] = 0
                continue

            # Filter executive transactions
            position_col = period_data['position'].fillna('').str.lower()
            exec_mask = position_col.str.contains('|'.join(executive_keywords), na=False)
            exec_data = period_data[exec_mask]

            if exec_data.empty:
                result.loc[ref_date, 'insider_exec_buy_count'] = 0
                result.loc[ref_date, 'insider_exec_sell_count'] = 0
            else:
                result.loc[ref_date, 'insider_exec_buy_count'] = (exec_data['transaction_type'] == 'buy').sum()
                result.loc[ref_date, 'insider_exec_sell_count'] = (exec_data['transaction_type'] == 'sell').sum()

        return result

    def extract_all(self, df: pd.DataFrame, symbol: str = None, insider_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all insider trading features

        Args:
            df: Price DataFrame (used for index alignment)
            symbol: Stock symbol (to load insider data)
            insider_df: Pre-loaded insider data (optional)

        Returns:
            DataFrame with all insider features aligned to price dates
        """
        if insider_df is None and symbol:
            insider_df = self.load_insider_data(symbol)

        if insider_df is None:
            insider_df = pd.DataFrame()

        ref_dates = df.index

        # Calculate all feature groups
        sentiment = self.calculate_insider_sentiment(insider_df, ref_dates)
        recency = self.calculate_insider_recency(insider_df, ref_dates)
        magnitude = self.calculate_insider_magnitude(insider_df, ref_dates)
        position = self.calculate_insider_position_features(insider_df, ref_dates)

        # Combine all features
        features = pd.concat([
            sentiment,
            recency,
            magnitude,
            position,
        ], axis=1)

        return features


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    np.random.seed(42)

    # Create sample insider data
    insider_data = []
    insider_dates = dates[::5]  # One transaction every 5 days
    for i, date in enumerate(insider_dates):
        insider_data.append({
            'date': date,
            'insider_name': f'Insider {i % 3}',
            'position': ['Director', 'Commissioner', 'CEO'][i % 3],
            'transaction_type': ['buy', 'sell'][i % 2],
            'shares': np.random.randint(1000, 100000),
            'price': np.random.randint(1000, 5000),
            'value': np.random.randint(1000000, 100000000),
        })

    insider_df = pd.DataFrame(insider_data)

    # Create sample price data
    price_df = pd.DataFrame({
        'close': 1000 + np.cumsum(np.random.randn(30) * 10),
        'volume': np.random.randint(100000, 1000000, 30),
    }, index=dates)

    # Extract features
    extractor = InsiderFeatures(lookback_days=30)
    features = extractor.extract_all(price_df, insider_df=insider_df)

    print("Insider Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data:")
    print(features.tail(10))
