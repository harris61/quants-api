"""
Broker Activity Features for ML Model
Broker concentration, smart money indicators, institutional flow patterns
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class BrokerFeatures:
    """Extract features from broker summary data"""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20]

    def load_broker_data(self, symbol: str, start_date: str = None) -> pd.DataFrame:
        """Load broker summary data from database"""
        from database import session_scope, get_stock_by_symbol
        from database.models import BrokerSummary

        with session_scope() as session:
            stock = get_stock_by_symbol(session, symbol)
            if not stock:
                return pd.DataFrame()

            query = session.query(BrokerSummary).filter(
                BrokerSummary.stock_id == stock.id
            )
            if start_date:
                query = query.filter(BrokerSummary.date >= start_date)

            records = query.order_by(BrokerSummary.date).all()

            if not records:
                return pd.DataFrame()

            data = [{
                'date': r.date,
                'broker_code': r.broker_code,
                'buy_value': r.buy_value or 0,
                'sell_value': r.sell_value or 0,
                'net_value': r.net_value or 0,
                'buy_volume': r.buy_volume or 0,
                'sell_volume': r.sell_volume or 0,
                'net_volume': r.net_volume or 0,
            } for r in records]

            return pd.DataFrame(data)

    def aggregate_daily_broker_stats(self, broker_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate broker data to daily level with key statistics"""
        if broker_df.empty:
            return pd.DataFrame()

        daily_stats = broker_df.groupby('date').agg({
            'buy_value': ['sum', 'count'],
            'sell_value': 'sum',
            'net_value': ['sum', 'mean', 'std'],
            'buy_volume': 'sum',
            'sell_volume': 'sum',
        })

        daily_stats.columns = [
            'broker_total_buy_value', 'broker_active_count',
            'broker_total_sell_value',
            'broker_total_net_value', 'broker_avg_net', 'broker_std_net',
            'broker_total_buy_volume', 'broker_total_sell_volume'
        ]

        daily_stats.index = pd.to_datetime(daily_stats.index)
        return daily_stats

    def calculate_broker_concentration(self, broker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate broker concentration metrics
        - Top 5 broker share of total volume
        - HHI (Herfindahl-Hirschman Index) for concentration
        """
        if broker_df.empty:
            return pd.DataFrame()

        results = []

        for date in broker_df['date'].unique():
            day_data = broker_df[broker_df['date'] == date].copy()
            total_volume = day_data['buy_volume'].sum() + day_data['sell_volume'].sum()

            if total_volume == 0:
                continue

            # Calculate broker market share
            day_data['total_volume'] = day_data['buy_volume'] + day_data['sell_volume']
            day_data['market_share'] = day_data['total_volume'] / total_volume

            # Top 5 broker concentration
            top5_share = day_data.nlargest(5, 'total_volume')['market_share'].sum()

            # HHI (sum of squared market shares * 10000)
            hhi = (day_data['market_share'] ** 2).sum() * 10000

            # Net buyers vs net sellers count
            net_buyers = (day_data['net_value'] > 0).sum()
            net_sellers = (day_data['net_value'] < 0).sum()

            results.append({
                'date': date,
                'broker_top5_concentration': top5_share,
                'broker_hhi': hhi,
                'broker_net_buyers_count': net_buyers,
                'broker_net_sellers_count': net_sellers,
                'broker_buyer_seller_ratio': net_buyers / (net_sellers + 1),
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_smart_money_indicators(self, broker_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate smart money indicators based on broker patterns
        - Large broker activity (institutional)
        - Broker accumulation/distribution
        """
        if broker_df.empty:
            return pd.DataFrame()

        results = []

        for date in broker_df['date'].unique():
            day_data = broker_df[broker_df['date'] == date].copy()

            # Large brokers (top quartile by volume)
            total_vol = day_data['buy_volume'] + day_data['sell_volume']
            if total_vol.sum() == 0:
                continue

            threshold = total_vol.quantile(0.75) if len(day_data) >= 4 else total_vol.median()
            large_brokers = day_data[total_vol >= threshold]
            small_brokers = day_data[total_vol < threshold]

            # Large broker net flow
            large_net = large_brokers['net_value'].sum()
            total_abs_net = day_data['net_value'].abs().sum()
            large_net_pct = large_net / (total_abs_net + 1) if total_abs_net > 0 else 0

            # Small broker net flow (retail indicator)
            small_net = small_brokers['net_value'].sum()

            # Smart money divergence (large vs small)
            smart_money_divergence = np.sign(large_net) - np.sign(small_net)

            results.append({
                'date': date,
                'broker_large_net': large_net,
                'broker_large_net_pct': large_net_pct,
                'broker_small_net': small_net,
                'broker_smart_money_divergence': smart_money_divergence,
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df.set_index('date', inplace=True)

        return result_df

    def calculate_broker_momentum(self, daily_stats: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling broker flow momentum"""
        if daily_stats.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=daily_stats.index)

        if 'broker_total_net_value' not in daily_stats.columns:
            return result

        net_value = daily_stats['broker_total_net_value']

        # Rolling net value sum and moving average
        for period in self.lookback_periods:
            result[f'broker_net_sum_{period}d'] = net_value.rolling(period, min_periods=1).sum()
            result[f'broker_net_ma_{period}d'] = net_value.rolling(period, min_periods=1).mean()

        # Net value change
        result['broker_net_change_1d'] = net_value.diff(1)
        result['broker_net_change_5d'] = net_value.diff(5)

        # Active broker trend
        if 'broker_active_count' in daily_stats.columns:
            result['broker_active_ma_5d'] = daily_stats['broker_active_count'].rolling(5, min_periods=1).mean()

        # Broker flow momentum (short vs long MA)
        if 'broker_net_ma_5d' in result.columns and 'broker_net_ma_20d' in result.columns:
            result['broker_flow_momentum'] = result['broker_net_ma_5d'] - result['broker_net_ma_20d']

        return result

    def extract_all(self, df: pd.DataFrame, symbol: str = None, broker_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all broker-based features

        Args:
            df: Price DataFrame (used for index alignment)
            symbol: Stock symbol (to load broker data)
            broker_df: Pre-loaded broker data (optional)

        Returns:
            DataFrame with all broker features aligned to price dates
        """
        if broker_df is None and symbol:
            broker_df = self.load_broker_data(symbol)

        if broker_df is None or broker_df.empty:
            # Return empty features with correct index
            return pd.DataFrame(index=df.index)

        # Calculate all feature groups
        daily_stats = self.aggregate_daily_broker_stats(broker_df)
        concentration = self.calculate_broker_concentration(broker_df)
        smart_money = self.calculate_smart_money_indicators(broker_df)
        momentum = self.calculate_broker_momentum(daily_stats)

        # Combine all features
        feature_dfs = [daily_stats, concentration, smart_money, momentum]
        features = pd.concat([f for f in feature_dfs if not f.empty], axis=1)

        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        # Align to price DataFrame index
        features = features.reindex(df.index)

        return features


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    np.random.seed(42)

    # Create sample broker data
    broker_data = []
    for date in dates:
        for broker in ['AA', 'BB', 'CC', 'DD', 'EE']:
            broker_data.append({
                'date': date,
                'broker_code': broker,
                'buy_value': np.random.randint(100000, 1000000),
                'sell_value': np.random.randint(100000, 1000000),
                'net_value': np.random.randint(-500000, 500000),
                'buy_volume': np.random.randint(10000, 100000),
                'sell_volume': np.random.randint(10000, 100000),
            })

    broker_df = pd.DataFrame(broker_data)

    # Create sample price data
    price_df = pd.DataFrame({
        'close': 1000 + np.cumsum(np.random.randn(10) * 10),
        'volume': np.random.randint(100000, 1000000, 10),
    }, index=dates)

    # Extract features
    extractor = BrokerFeatures()
    features = extractor.extract_all(price_df, broker_df=broker_df)

    print("Broker Features Shape:", features.shape)
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nSample data:")
    print(features.head())
