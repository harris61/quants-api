"""
Backtester - Walk-forward backtesting for model evaluation
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from database import session_scope, Stock, DailyPrice, DailyMover
from features.pipeline import FeaturePipeline
from models.trainer import ModelTrainer
from config import (
    BACKTEST_TRAIN_DAYS, BACKTEST_TEST_DAYS,
    TOP_PICKS_COUNT, MODELS_DIR
)
from config import MOVERS_FILTER_ENABLED, MOVERS_FILTER_TYPES


class Backtester:
    """Walk-forward backtesting for strategy evaluation"""

    def __init__(
        self,
        train_days: int = None,
        test_days: int = None,
        top_k: int = None
    ):
        self.train_days = train_days or BACKTEST_TRAIN_DAYS
        self.test_days = test_days or BACKTEST_TEST_DAYS
        self.top_k = top_k or TOP_PICKS_COUNT
        self.pipeline = FeaturePipeline()
        self.results = []
        self.pick_results = []

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get available date range from database"""
        with session_scope() as session:
            min_date = session.query(DailyPrice.date).order_by(
                DailyPrice.date.asc()
            ).first()
            max_date = session.query(DailyPrice.date).order_by(
                DailyPrice.date.desc()
            ).first()

            return min_date[0], max_date[0]

    def build_dataset_for_period(
        self,
        start_date: str,
        end_date: str,
        min_samples: int = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build dataset for a specific period"""
        X, y = self.pipeline.build_training_dataset(
            start_date=start_date,
            end_date=end_date,
            min_samples=min_samples,
            show_progress=False
        )
        return X, y

    def get_trading_dates(self, start_date: str, end_date: str) -> List[datetime]:
        """Get sorted list of trading dates within a range"""
        with session_scope() as session:
            rows = session.query(DailyPrice.date).filter(
                DailyPrice.date >= start_date,
                DailyPrice.date <= end_date
            ).distinct().order_by(DailyPrice.date.asc()).all()
        return [r[0] for r in rows]

    def build_next_date_map(self, dates: List[datetime]) -> Dict[datetime, datetime]:
        """Map each trading date to the next trading date"""
        next_map = {}
        for i in range(len(dates) - 1):
            next_map[dates[i]] = dates[i + 1]
        return next_map

    def build_cooldown_map(self, dates: List[datetime], cooldown_days: int = 3) -> Dict[datetime, Optional[datetime]]:
        """Map each trade date to the next eligible trade date after settlement"""
        next_trade = {}
        offset = cooldown_days + 1
        for i, date in enumerate(dates):
            next_trade[date] = dates[i + offset] if i + offset < len(dates) else None
        return next_trade

    def load_returns_lookup(self, symbols: List[str], dates: List[datetime]) -> Dict[Tuple[str, datetime], float]:
        """Load actual intraday returns (open -> close) for symbols on specific dates"""
        if not symbols or not dates:
            return {}

        with session_scope() as session:
            rows = session.query(Stock.symbol, DailyPrice.date, DailyPrice.open, DailyPrice.close).join(
                DailyPrice, DailyPrice.stock_id == Stock.id
            ).filter(
                Stock.symbol.in_(symbols),
                DailyPrice.date.in_(dates)
            ).all()

        lookup = {}
        for symbol, date, open_price, close_price in rows:
            if open_price is None or close_price is None or open_price == 0:
                continue
            lookup[(symbol, date)] = (close_price - open_price) / open_price

        return lookup

    def load_movers_lookup(self, start_date: str, end_date: str) -> Dict[datetime, set]:
        """Load movers symbols per date for filtering"""
        if not MOVERS_FILTER_ENABLED:
            return {}
        with session_scope() as session:
            rows = session.query(DailyMover.date, Stock.symbol).join(
                Stock, Stock.id == DailyMover.stock_id
            ).filter(
                DailyMover.date >= start_date,
                DailyMover.date <= end_date,
                DailyMover.mover_type.in_(MOVERS_FILTER_TYPES)
            ).all()
        movers = {}
        for date, symbol in rows:
            movers.setdefault(date, set()).add(symbol)
        return movers

    def run_walk_forward(
        self,
        start_date: str = None,
        end_date: str = None,
        step_days: int = None
    ) -> pd.DataFrame:
        """
        Run walk-forward backtesting

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            step_days: Days to step forward each iteration

        Returns:
            DataFrame with backtest results
        """
        step_days = step_days or self.test_days

        # Get date range
        min_date, max_date = self.get_date_range()

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            start_dt = min_date + timedelta(days=self.train_days)

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            end_dt = max_date

        print(f"Backtesting from {start_dt} to {end_dt}")
        print(f"Training window: {self.train_days} days")
        print(f"Test window: {self.test_days} days")
        print(f"Step size: {step_days} days")
        print("=" * 60)

        current_date = start_dt
        self.results = []
        self.pick_results = []

        while current_date + timedelta(days=self.test_days) <= end_dt:
            # Define periods
            train_start = (current_date - timedelta(days=self.train_days)).strftime("%Y-%m-%d")
            train_end = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
            test_start = current_date.strftime("%Y-%m-%d")
            test_end = (current_date + timedelta(days=self.test_days - 1)).strftime("%Y-%m-%d")

            print(f"\nPeriod: Train [{train_start} to {train_end}] -> Test [{test_start} to {test_end}]")

            try:
                # Build training data
                X_train, y_train = self.build_dataset_for_period(train_start, train_end)

                if X_train.empty or len(X_train) < 100:
                    print(f"  Insufficient training data, skipping...")
                    current_date += timedelta(days=step_days)
                    continue

                # Prepare features
                X_train_clean, y_train_clean, feature_names = self.pipeline.prepare_for_training(
                    X_train, y_train
                )

                # Train model
                trainer = ModelTrainer()
                trainer.feature_names = feature_names

                # Quick training for backtest
                params = trainer.get_default_params()
                params['n_estimators'] = 200
                params['early_stopping_rounds'] = 30

                # Create a small validation split for early stopping
                stratify = y_train_clean if len(np.unique(y_train_clean)) > 1 else None
                if len(y_train_clean) >= 200:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train_clean,
                        y_train_clean,
                        test_size=0.1,
                        random_state=42,
                        stratify=stratify
                    )
                    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
                    valid_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)
                    trainer.model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[train_data, valid_data],
                        valid_names=['train', 'valid'],
                        num_boost_round=200,
                        callbacks=[lgb.log_evaluation(0)]
                    )
                else:
                    params.pop('early_stopping_rounds', None)
                    train_data = lgb.Dataset(X_train_clean, label=y_train_clean, feature_name=feature_names)
                    trainer.model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,
                        callbacks=[lgb.log_evaluation(0)]
                    )

                # Build test data with lookback buffer for feature calculation
                lookback_days = 60
                lookback_start = (datetime.strptime(test_start, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                X_test, y_test = self.build_dataset_for_period(
                    lookback_start,
                    test_end,
                    min_samples=1
                )

                if X_test.empty:
                    print(f"  No test data, skipping...")
                    current_date += timedelta(days=step_days)
                    continue
                # Keep only rows within the test window
                test_start_dt = pd.to_datetime(test_start)
                test_end_dt = pd.to_datetime(test_end)
                mask = (X_test.index >= test_start_dt) & (X_test.index <= test_end_dt)
                X_test = X_test.loc[mask]
                y_test = y_test.loc[mask]

                if X_test.empty:
                    print(f"  No test data in window, skipping...")
                    current_date += timedelta(days=step_days)
                    continue

                # Get symbols for test period
                if 'symbol' in X_test.columns:
                    test_symbols = X_test['symbol'].tolist()
                else:
                    test_symbols = ['unknown'] * len(X_test)

                # Prepare test features
                X_test_clean = X_test.select_dtypes(include=[np.number])
                for col in feature_names:
                    if col not in X_test_clean.columns:
                        X_test_clean[col] = 0
                X_test_clean = X_test_clean[feature_names]
                X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

                # Make predictions
                probabilities = trainer.model.predict(X_test_clean.values)

                # Create test results with date and symbol
                test_results = pd.DataFrame({
                    'date': X_test.index,
                    'symbol': test_symbols,
                    'probability': probabilities,
                    'actual_label': y_test.values
                })

                # Build target date lookup and returns
                trading_dates = self.get_trading_dates(test_start, test_end)
                next_date_map = self.build_next_date_map(trading_dates)
                target_dates = list(set(next_date_map.values()))
                returns_lookup = self.load_returns_lookup(
                    symbols=list(set(test_results['symbol'].tolist())),
                    dates=target_dates
                )
                movers_lookup = self.load_movers_lookup(test_start, test_end)

                # Evaluate on each day in test period
                daily_results, pick_results = self._evaluate_period(
                    test_results,
                    test_start,
                    test_end,
                    next_date_map,
                    returns_lookup,
                    trading_dates,
                    movers_lookup
                )

                self.results.extend(daily_results)
                self.pick_results.extend(pick_results)

            except Exception as e:
                print(f"  Error: {e}")

            current_date += timedelta(days=step_days)

        # Compile results
        if self.results:
            results_df = pd.DataFrame(self.results)
            self._print_summary(results_df)
            return results_df
        else:
            print("No backtest results generated!")
            return pd.DataFrame()

    def _evaluate_period(
        self,
        test_results: pd.DataFrame,
        start_date: str,
        end_date: str,
        next_date_map: Dict[datetime, datetime],
        returns_lookup: Dict[Tuple[str, datetime], float],
        trading_dates: List[datetime],
        movers_lookup: Dict[datetime, set]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Evaluate predictions for each day in the test period"""
        daily_results = []
        pick_results = []

        test_results['date'] = pd.to_datetime(test_results['date']).dt.date
        unique_dates = sorted(test_results['date'].unique())
        cooldown_map = self.build_cooldown_map(trading_dates, cooldown_days=3)
        next_allowed_trade_date = trading_dates[0] if trading_dates else None

        for date in unique_dates:
            target_date = next_date_map.get(date)
            if target_date is None or next_allowed_trade_date is None:
                continue
            if target_date < next_allowed_trade_date:
                continue

            day_results = test_results[test_results['date'] == date]
            if day_results.empty:
                continue

            # Optional movers filter
            if MOVERS_FILTER_ENABLED:
                mover_symbols = movers_lookup.get(date, set())
                if mover_symbols:
                    day_results = day_results[day_results['symbol'].isin(mover_symbols)]
                    if day_results.empty:
                        continue

            day_results = day_results.sort_values('probability', ascending=False)
            top_picks = day_results.head(self.top_k).copy()

            total_predictions = len(top_picks)
            correct_predictions = int(top_picks['actual_label'].sum())
            precision = correct_predictions / total_predictions if total_predictions > 0 else 0
            avg_probability = top_picks['probability'].mean() if total_predictions > 0 else 0

            # Record pick-level results
            for rank, (_, row) in enumerate(top_picks.iterrows(), start=1):
                actual_return = returns_lookup.get((row['symbol'], target_date))
                pick_results.append({
                    'date': date,
                    'target_date': target_date,
                    'symbol': row['symbol'],
                    'probability': row['probability'],
                    'rank': rank,
                    'actual_return': actual_return,
                    'is_top_gainer': bool(row['actual_label']),
                    'is_correct': bool(row['actual_label']),
                })

            daily_results.append({
                'test_date': date,
                'target_date': target_date,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'precision_at_k': precision,
                'avg_probability': avg_probability,
            })

            print(f"  {date} Precision@{self.top_k}: {precision:.4f} ({correct_predictions}/{total_predictions})")
            next_allowed_trade_date = cooldown_map.get(target_date)
            if next_allowed_trade_date is None:
                break

        return daily_results, pick_results

    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print backtest summary"""
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        print(f"\nTotal trade days: {len(results_df)}")
        print(f"Average Precision@{self.top_k}: {results_df['precision_at_k'].mean():.4f}")
        print(f"Std Precision@{self.top_k}: {results_df['precision_at_k'].std():.4f}")
        print(f"Min Precision@{self.top_k}: {results_df['precision_at_k'].min():.4f}")
        print(f"Max Precision@{self.top_k}: {results_df['precision_at_k'].max():.4f}")

        # Hit rate (% of days with at least 1 correct)
        hit_rate = (results_df['correct_predictions'] > 0).mean()
        print(f"Hit Rate: {hit_rate:.4f} ({(results_df['correct_predictions'] > 0).sum()}/{len(results_df)})")

        # Total correct predictions
        total_correct = results_df['correct_predictions'].sum()
        total_preds = results_df['total_predictions'].sum()
        overall_precision = total_correct / total_preds if total_preds > 0 else 0
        print(f"Overall Precision: {overall_precision:.4f} ({total_correct}/{total_preds})")

    def run_simple_backtest(
        self,
        model_name: str = None,
        test_days: int = 30
    ) -> pd.DataFrame:
        """
        Run simple backtest on last N days using existing model

        Args:
            model_name: Name of model to use
            test_days: Number of days to backtest

        Returns:
            DataFrame with daily results
        """
        from models.predictor import Predictor

        predictor = Predictor(model_name=model_name)

        # Get date range
        _, max_date = self.get_date_range()
        end_date = max_date
        start_date = max_date - timedelta(days=test_days)

        results = []

        # For each day in test period
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")

            # Get features for this date
            features = self.pipeline.get_latest_features()

            if features.empty:
                current_date += timedelta(days=1)
                continue

            # Make predictions
            probabilities = predictor.predict_proba(features)

            # Get actual next day returns
            # (would need to implement this properly with actual data)

            current_date += timedelta(days=1)

        return pd.DataFrame(results)

    def save_results(self, results_df: pd.DataFrame, filename: str = None) -> str:
        """Save backtest results to CSV"""
        if filename is None:
            filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = os.path.join(MODELS_DIR, filename)
        results_df.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")

        if self.pick_results:
            picks_df = pd.DataFrame(self.pick_results)
            if not picks_df.empty:
                picks_df = picks_df.sort_values(['date', 'rank'])
                picks_filename = filename.replace(".csv", "_picks.csv")
                picks_path = os.path.join(MODELS_DIR, picks_filename)
                picks_df.to_csv(picks_path, index=False)
                print(f"Pick details saved to: {picks_path}")

        return filepath


def run_backtest():
    """CLI function to run backtest"""
    import argparse

    parser = argparse.ArgumentParser(description="Run walk-forward backtest")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--train-days", type=int, default=BACKTEST_TRAIN_DAYS,
                        help="Training window in days")
    parser.add_argument("--test-days", type=int, default=BACKTEST_TEST_DAYS,
                        help="Test window in days")
    parser.add_argument("--top-k", type=int, default=TOP_PICKS_COUNT,
                        help="Number of top predictions")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    args = parser.parse_args()

    backtester = Backtester(
        train_days=args.train_days,
        test_days=args.test_days,
        top_k=args.top_k
    )

    results = backtester.run_walk_forward(
        start_date=args.start,
        end_date=args.end
    )

    if args.save and not results.empty:
        backtester.save_results(results)


if __name__ == "__main__":
    run_backtest()
