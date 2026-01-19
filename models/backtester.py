"""
Backtester - Walk-forward backtesting for model evaluation
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from database import session_scope, Stock, DailyPrice
from features.pipeline import FeaturePipeline
from models.trainer import ModelTrainer
from config import (
    BACKTEST_TRAIN_DAYS, BACKTEST_TEST_DAYS,
    TOP_PICKS_COUNT, TOP_GAINER_THRESHOLD, MODELS_DIR
)


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
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build dataset for a specific period"""
        X, y = self.pipeline.build_training_dataset(
            start_date=start_date,
            end_date=end_date,
            show_progress=False
        )
        return X, y

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

                train_data = lgb.Dataset(X_train_clean, label=y_train_clean, feature_name=feature_names)
                trainer.model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,
                    callbacks=[lgb.log_evaluation(0)]
                )

                # Build test data
                X_test, y_test = self.build_dataset_for_period(test_start, test_end)

                if X_test.empty:
                    print(f"  No test data, skipping...")
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

                # Create test results
                test_results = pd.DataFrame({
                    'symbol': test_symbols,
                    'probability': probabilities,
                    'actual_label': y_test.values
                })

                # Evaluate on each day in test period
                period_results = self._evaluate_period(
                    test_results,
                    test_start,
                    test_end
                )

                self.results.extend(period_results)

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
        end_date: str
    ) -> List[Dict]:
        """Evaluate predictions for a test period"""
        results = []

        # Sort by probability and get top K
        test_results = test_results.sort_values('probability', ascending=False)
        top_picks = test_results.head(self.top_k)

        # Calculate metrics
        total_predictions = len(top_picks)
        correct_predictions = top_picks['actual_label'].sum()
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Average probability of picks
        avg_probability = top_picks['probability'].mean()

        results.append({
            'test_start': start_date,
            'test_end': end_date,
            'total_predictions': total_predictions,
            'correct_predictions': int(correct_predictions),
            'precision_at_k': precision,
            'avg_probability': avg_probability,
        })

        print(f"  Precision@{self.top_k}: {precision:.4f} ({int(correct_predictions)}/{total_predictions})")

        return results

    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print backtest summary"""
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        print(f"\nTotal test periods: {len(results_df)}")
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
