"""
Feature Pipeline - Combine all features into training dataset
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from database import session_scope, Stock, DailyPrice
from features.price_features import PriceFeatures
from features.volume_features import VolumeFeatures
from features.foreign_features import ForeignFlowFeatures
from features.technical import TechnicalFeatures
from features.market_features import MarketRegimeFeatures
from features.broker_features import BrokerFeatures
from features.insider_features import InsiderFeatures
from features.intraday_features import IntradayFeatures
from features.mover_features import MoverFeatures
from config import (
    TOP_GAINER_THRESHOLD,
    MIN_TRAINING_SAMPLES,
    EQUITY_SYMBOL_REGEX,
    INCLUDE_MOVER_FEATURES,
)


class FeaturePipeline:
    """Pipeline to extract features and create training dataset"""

    def __init__(
        self,
        include_broker: bool = True,
        include_insider: bool = True,
        include_intraday: bool = True,
        include_movers: bool = True
    ):
        # Core extractors (always included)
        self.price_extractor = PriceFeatures()
        self.volume_extractor = VolumeFeatures()
        self.foreign_extractor = ForeignFlowFeatures()
        self.technical_extractor = TechnicalFeatures()
        self.market_extractor = MarketRegimeFeatures()

        # Optional extractors (new data sources)
        self.include_broker = include_broker
        self.include_insider = include_insider
        self.include_intraday = include_intraday
        self.include_movers = include_movers and INCLUDE_MOVER_FEATURES

        if include_broker:
            self.broker_extractor = BrokerFeatures()
        if include_insider:
            self.insider_extractor = InsiderFeatures()
        if include_intraday:
            self.intraday_extractor = IntradayFeatures()
        if self.include_movers:
            self.mover_extractor = MoverFeatures()
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def is_equity_symbol(self, symbol: str) -> bool:
        """Return True for normal equity tickers"""
        if not symbol:
            return False
        return bool(self._equity_pattern.match(symbol.upper()))

    def load_all_stock_data_batch(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for multiple symbols in a single batch query.
        Much more efficient than loading one stock at a time.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping symbol -> DataFrame with OHLCV data
        """
        if not symbols:
            return {}

        with session_scope() as session:
            query = session.query(
                Stock.symbol,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.value,
                DailyPrice.frequency,
                DailyPrice.foreign_buy,
                DailyPrice.foreign_sell,
                DailyPrice.foreign_net,
            ).join(DailyPrice, DailyPrice.stock_id == Stock.id).filter(
                Stock.symbol.in_([s.upper() for s in symbols])
            )

            if start_date:
                query = query.filter(DailyPrice.date >= start_date)
            if end_date:
                query = query.filter(DailyPrice.date <= end_date)

            query = query.order_by(Stock.symbol, DailyPrice.date)
            rows = query.all()

        if not rows:
            return {}

        # Convert to DataFrame
        columns = [
            'symbol', 'date', 'open', 'high', 'low', 'close',
            'volume', 'value', 'frequency', 'foreign_buy', 'foreign_sell', 'foreign_net'
        ]
        df_all = pd.DataFrame(rows, columns=columns)

        # Convert types
        df_all['date'] = pd.to_datetime(df_all['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'frequency',
                        'foreign_buy', 'foreign_sell', 'foreign_net']
        for col in numeric_cols:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

        # Group by symbol and return dict
        result = {}
        for symbol, group in df_all.groupby('symbol'):
            stock_df = group.drop('symbol', axis=1).set_index('date').sort_index()
            result[symbol] = stock_df

        return result

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
            self.market_extractor.extract_all(df),
        ]

        # Optional feature extractors (require symbol for data loading)
        if symbol:
            optional_extractors = [
                ("broker", self.include_broker, getattr(self, "broker_extractor", None)),
                ("insider", self.include_insider, getattr(self, "insider_extractor", None)),
                ("intraday", self.include_intraday, getattr(self, "intraday_extractor", None)),
                ("movers", self.include_movers, getattr(self, "mover_extractor", None)),
            ]

            for _, enabled, extractor in optional_extractors:
                if not enabled or extractor is None:
                    continue
                try:
                    extracted = extractor.extract_all(df, symbol=symbol)
                    if not extracted.empty:
                        feature_dfs.append(extracted)
                except Exception:
                    pass  # Silently skip if optional data not available

        # Combine all features
        features = pd.concat(feature_dfs, axis=1)

        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        # Align features to be strictly prior-day data
        features = features.shift(1)

        return features

    def create_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Create next-day daily return series (close[t] -> close[t+1]).

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series of next-day returns
        """
        return df['close'].pct_change().shift(-1)

    def create_labels(self, df: pd.DataFrame, threshold: float = None) -> pd.Series:
        """
        Create labels for next-day daily return prediction (prev close -> close)

        Args:
            df: DataFrame with 'open' and 'close' columns
            threshold: Return threshold for positive label (default: TOP_GAINER_THRESHOLD)

        Returns:
            Series with binary labels (1 = top gainer potential, 0 = not)
        """
        threshold = threshold or TOP_GAINER_THRESHOLD

        # Next-day daily return (close[t] -> close[t+1])
        next_day_return = self.create_returns(df)

        # Binary label
        labels = (next_day_return >= threshold).astype(int)

        return labels

    def build_dataset_for_stock(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        label_type: str = "binary"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature dataset for a single stock

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (features DataFrame, labels/returns Series)
        """
        # Load data
        df = self.load_stock_data(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Extract features (pass symbol for new extractors)
        features = self.extract_features_for_stock(df, symbol=symbol)
        if features.empty:
            return pd.DataFrame(), pd.Series()

        # Create labels or returns
        if label_type == "return":
            labels = self.create_returns(df)
        else:
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
        show_progress: bool = True,
        label_type: str = "binary",
        use_batch_loading: bool = True,
        include_delisted: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset for multiple stocks

        Args:
            symbols: List of stock symbols. If None, uses all stocks in DB.
            start_date: Start date
            end_date: End date
            min_samples: Minimum samples required per stock
            show_progress: Show progress bar
            label_type: "binary" for classification, "return" for ranking
            use_batch_loading: Use efficient batch DB query (recommended)
            include_delisted: Include inactive/delisted stocks to avoid survivorship bias

        Returns:
            Tuple of (features DataFrame, labels/returns Series)
        """
        min_samples = min_samples or MIN_TRAINING_SAMPLES

        # Get symbols from database if not provided
        if symbols is None:
            with session_scope() as session:
                if include_delisted:
                    # Include all stocks (active and inactive) to avoid survivorship bias
                    stocks = session.query(Stock).all()
                else:
                    # Only active stocks (default, for prediction)
                    stocks = session.query(Stock).filter(Stock.is_active == True).all()
                symbols = [s.symbol for s in stocks if self.is_equity_symbol(s.symbol)]

        if not symbols:
            print("No symbols found!")
            return pd.DataFrame(), pd.Series()

        # Filter to equity symbols only
        symbols = [s for s in symbols if self.is_equity_symbol(s)]

        all_features = []
        all_labels = []

        # Batch load all stock data in single query (much faster)
        if use_batch_loading:
            if show_progress:
                print(f"Batch loading data for {len(symbols)} stocks...")
            stock_data = self.load_all_stock_data_batch(symbols, start_date, end_date)
            if show_progress:
                print(f"Loaded data for {len(stock_data)} stocks")

            iterator = tqdm(symbols, desc="Extracting features") if show_progress else symbols

            for symbol in iterator:
                try:
                    df = stock_data.get(symbol)
                    if df is None or df.empty or len(df) < 50:
                        continue

                    # Extract features
                    features = self.extract_features_for_stock(df, symbol=symbol)
                    if features.empty:
                        continue

                    # Create labels or returns
                    if label_type == "return":
                        labels = self.create_returns(df)
                    else:
                        labels = self.create_labels(df)

                    # Add symbol column
                    features['symbol'] = symbol

                    if len(features) >= min_samples:
                        all_features.append(features)
                        all_labels.append(labels)

                except Exception as e:
                    if show_progress:
                        print(f"Error processing {symbol}: {e}")
                    continue
        else:
            # Legacy: load one stock at a time
            iterator = tqdm(symbols, desc="Building dataset") if show_progress else symbols

            for symbol in iterator:
                try:
                    features, labels = self.build_dataset_for_stock(
                        symbol, start_date, end_date, label_type=label_type
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
                symbols = [s.symbol for s in stocks if self.is_equity_symbol(s.symbol)]

        all_features = []
        iterator = tqdm(symbols, desc="Extracting latest features") if show_progress else symbols

        for symbol in iterator:
            if not self.is_equity_symbol(symbol):
                continue
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
