"""
Feature Pipeline - Combine all features into training dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from database import session_scope, Stock, DailyPrice
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures
from features.foreign_features import ForeignFlowFeatures
from features.technical import TechnicalFeatures
from features.broker_features import BrokerFeatures
from features.insider_features import InsiderFeatures
from features.intraday_features import IntradayFeatures
from config import TOP_GAINER_THRESHOLD, MIN_TRAINING_SAMPLES


class FeaturePipeline:
    """Pipeline to extract features and create training dataset"""

    def __init__(
        self,
        include_broker: bool = True,
        include_insider: bool = True,
        include_intraday: bool = True
    ):
        # Core extractors (always included)
        self.price_extractor = PriceFeatures()
        self.volume_extractor = VolumeFeatures()
        self.foreign_extractor = ForeignFlowFeatures()
        self.technical_extractor = TechnicalFeatures()

        # Optional extractors (new data sources)
        self.include_broker = include_broker
        self.include_insider = include_insider
        self.include_intraday = include_intraday

        if include_broker:
            self.broker_extractor = BrokerFeatures()
        if include_insider:
            self.insider_extractor = InsiderFeatures()
        if include_intraday:
            self.intraday_extractor = IntradayFeatures()

    def load_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load stock data from database

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV and foreign flow data
        """
        with session_scope() as session:
            query = session.query(DailyPrice).join(Stock).filter(
                Stock.symbol == symbol.upper()
            )

            if start_date:
                query = query.filter(DailyPrice.date >= start_date)
            if end_date:
                query = query.filter(DailyPrice.date <= end_date)

            query = query.order_by(DailyPrice.date)
            records = query.all()

            if not records:
                return pd.DataFrame()

            data = []
            for r in records:
                data.append({
                    'date': r.date,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.volume,
                    'value': r.value,
                    'frequency': r.frequency,
                    'foreign_buy': r.foreign_buy,
                    'foreign_sell': r.foreign_sell,
                    'foreign_net': r.foreign_net,
                })

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            numeric_cols = [
                'open', 'high', 'low', 'close', 'volume', 'value', 'frequency',
                'foreign_buy', 'foreign_sell', 'foreign_net'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

    def extract_features_for_stock(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Extract all features for a single stock

        Args:
            df: DataFrame with OHLCV and foreign flow data
            symbol: Stock symbol (needed for broker/insider/intraday data)

        Returns:
            DataFrame with all features
        """
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        # Extract features from core extractors
        feature_dfs = [
            self.price_extractor.extract_all(df),
            self.volume_extractor.extract_all(df),
            self.foreign_extractor.extract_all(df),
            self.technical_extractor.extract_all(df),
        ]

        # New feature extractors (require symbol for data loading)
        if symbol:
            if self.include_broker and hasattr(self, 'broker_extractor'):
                try:
                    broker_features = self.broker_extractor.extract_all(df, symbol=symbol)
                    if not broker_features.empty:
                        feature_dfs.append(broker_features)
                except Exception as e:
                    pass  # Silently skip if broker data not available

            if self.include_insider and hasattr(self, 'insider_extractor'):
                try:
                    insider_features = self.insider_extractor.extract_all(df, symbol=symbol)
                    if not insider_features.empty:
                        feature_dfs.append(insider_features)
                except Exception as e:
                    pass  # Silently skip if insider data not available

            if self.include_intraday and hasattr(self, 'intraday_extractor'):
                try:
                    intraday_features = self.intraday_extractor.extract_all(df, symbol=symbol)
                    if not intraday_features.empty:
                        feature_dfs.append(intraday_features)
                except Exception as e:
                    pass  # Silently skip if intraday data not available

        # Combine all features
        features = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        return features

    def create_labels(self, df: pd.DataFrame, threshold: float = None) -> pd.Series:
        """
        Create labels for next-day return prediction

        Args:
            df: DataFrame with 'close' column
            threshold: Return threshold for positive label (default: TOP_GAINER_THRESHOLD)

        Returns:
            Series with binary labels (1 = top gainer potential, 0 = not)
        """
        threshold = threshold or TOP_GAINER_THRESHOLD

        # Next day return
        next_day_return = df['close'].pct_change().shift(-1)

        # Binary label
        labels = (next_day_return >= threshold).astype(int)

        return labels

    def build_dataset_for_stock(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature dataset for a single stock

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Load data
        df = self.load_stock_data(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Extract features (pass symbol for new extractors)
        features = self.extract_features_for_stock(df, symbol=symbol)
        if features.empty:
            return pd.DataFrame(), pd.Series()

        # Create labels
        labels = self.create_labels(df)

        # Add symbol column
        features['symbol'] = symbol

        return features, labels

    def build_training_dataset(
        self,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        min_samples: int = None,
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all stocks in DB.
            start_date: Start date
            end_date: End date
            min_samples: Minimum samples required per stock
            show_progress: Show progress bar

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        min_samples = min_samples or MIN_TRAINING_SAMPLES

        # Get symbols from database if not provided
        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        if not symbols:
            print("No symbols found!")
            return pd.DataFrame(), pd.Series()

        all_features = []
        all_labels = []

        iterator = tqdm(symbols, desc="Building dataset") if show_progress else symbols

        for symbol in iterator:
            try:
                features, labels = self.build_dataset_for_stock(
                    symbol, start_date, end_date
                )

                if len(features) >= min_samples:
                    all_features.append(features)
                    all_labels.append(labels)

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            print("No valid data found!")
            return pd.DataFrame(), pd.Series()

        # Combine all stocks
        X = pd.concat(all_features, ignore_index=False)
        y = pd.concat(all_labels, ignore_index=False)

        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Drop rows with NaN labels only; features will be imputed later
        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        print(f"\nDataset built:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {len(X.columns) - 1}")  # -1 for symbol column
        print(f"  Positive labels: {y.sum()} ({y.mean()*100:.2f}%)")

        return X, y

    def prepare_for_training(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare dataset for model training

        Args:
            X: Features DataFrame
            y: Labels Series

        Returns:
            Tuple of (cleaned features, labels, feature names)
        """
        # Remove symbol column for training
        if 'symbol' in X.columns:
            X_train = X.drop('symbol', axis=1)
        else:
            X_train = X.copy()

        # Get feature names
        feature_names = X_train.columns.tolist()

        # Replace inf with nan
        X_train = X_train.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with median
        X_train = X_train.fillna(X_train.median())

        return X_train, y, feature_names

    def get_latest_features(
        self,
        symbols: List[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Get features for the latest date (for prediction)

        Args:
            symbols: List of stock symbols
            show_progress: Show progress bar

        Returns:
            DataFrame with latest features for each stock
        """
        if symbols is None:
            with session_scope() as session:
                stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks]

        all_features = []
        iterator = tqdm(symbols, desc="Extracting latest features") if show_progress else symbols

        for symbol in iterator:
            try:
                # Load recent data (need enough for feature calculation)
                df = self.load_stock_data(symbol)

                if df.empty or len(df) < 50:
                    continue

                # Extract features (pass symbol for new extractors)
                features = self.extract_features_for_stock(df, symbol=symbol)

                if features.empty:
                    continue

                # Get only the latest row
                latest = features.iloc[[-1]].copy()
                latest['symbol'] = symbol
                latest['date'] = df.index[-1]

                all_features.append(latest)

            except Exception as e:
                continue

        if not all_features:
            return pd.DataFrame()

        result = pd.concat(all_features, ignore_index=True)
        return result


def build_dataset():
    """CLI function to build training dataset"""
    import argparse

    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="dataset.csv", help="Output file")
    args = parser.parse_args()

    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset(start_date=args.start, end_date=args.end)

    if not X.empty:
        # Combine features and labels
        dataset = X.copy()
        dataset['label'] = y
        dataset.to_csv(args.output)
        print(f"\nDataset saved to {args.output}")


if __name__ == "__main__":
    build_dataset()
