"""
Predictor - Load model and make predictions
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

from database import session_scope, Stock, DailyPrice, Prediction
from features.pipeline import FeaturePipeline
from config import (
    MODELS_DIR,
    TOP_PICKS_COUNT,
    TOP_GAINER_THRESHOLD,
    MOVERS_FILTER_ENABLED,
    MOVERS_FILTER_TYPES,
)


class Predictor:
    """Load trained model and make predictions"""

    def __init__(self, model_name: str = None, model_path: str = None):
        self.model = None
        self.feature_names = None
        self.feature_medians = None
        self.model_type = "classification"
        self.model_name = model_name
        self.model_path = model_path or MODELS_DIR
        self.pipeline = FeaturePipeline()

        if model_name:
            self.load(model_name)

    def load(self, model_name: str = None) -> None:
        """
        Load model from file

        Args:
            model_name: Model name (without extension)
        """
        if model_name:
            self.model_name = model_name

        if not self.model_name:
            # Find latest model
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.txt')]
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.model_path}")
            self.model_name = sorted(model_files)[-1].replace('.txt', '')

        # Load model
        model_file = os.path.join(self.model_path, f"{self.model_name}.txt")
        self.model = lgb.Booster(model_file=model_file)

        # Load metadata
        metadata_file = os.path.join(self.model_path, f"{self.model_name}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata.get('feature_names', [])
            self.feature_medians = metadata.get('feature_medians', {})
            self.model_type = metadata.get('model_type', 'classification')

        print(f"Model loaded: {self.model_name}")

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction

        Args:
            X: Features DataFrame

        Returns:
            Numpy array with aligned features
        """
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])

        # Align columns with training features
        if self.feature_names:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in X_numeric.columns:
                    X_numeric[col] = self.feature_medians.get(col, 0) if self.feature_medians else 0

            # Select only features used in training
            X_numeric = X_numeric[self.feature_names]

        # Replace inf and fill NaN
        X_clean = X_numeric.replace([np.inf, -np.inf], np.nan)
        if self.feature_medians:
            # Fill using stored medians from training
            X_clean = X_clean.fillna(self.feature_medians)
        # Fill any remaining NaN with column median (edge case for new features)
        # Then fill remaining with 0 (for completely missing columns)
        X_clean = X_clean.fillna(X_clean.median()).fillna(0)

        return X_clean.values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction scores or probabilities

        Args:
            X: Features DataFrame

        Returns:
            Array of model scores
        """
        X_prepared = self.prepare_features(X)
        return self.model.predict(X_prepared)

    def predict(
        self,
        symbols: List[str] = None,
        top_k: int = None,
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for stocks

        Args:
            symbols: List of stock symbols (None = all active stocks)
            top_k: Number of top predictions to return
            save_to_db: Save predictions to database

        Returns:
            DataFrame with predictions
        """
        top_k = top_k or TOP_PICKS_COUNT

        # Get latest features
        print("Extracting features...")
        features_df = self.pipeline.get_latest_features(symbols)

        if features_df.empty:
            print("No features available!")
            return pd.DataFrame()

        # Store symbol and date info
        symbols_list = features_df['symbol'].tolist()
        dates = features_df['date'].tolist() if 'date' in features_df.columns else [datetime.now().date()] * len(symbols_list)

        # Get probabilities
        print("Making predictions...")
        probabilities = self.predict_proba(features_df)

        # Create results DataFrame
        results = pd.DataFrame({
            'symbol': symbols_list,
            'probability': probabilities,
            'date': dates,
        })

        # Sort by probability
        results = results.sort_values('probability', ascending=False)
        results['rank'] = range(1, len(results) + 1)

        # Optional movers-based filter (trade only in mover lists)
        if MOVERS_FILTER_ENABLED:
            mover_date = results['date'].iloc[0] if not results.empty else None
            if mover_date is not None:
                mover_symbols = self._get_mover_symbols(mover_date, MOVERS_FILTER_TYPES)
                if mover_symbols:
                    results = results[results['symbol'].isin(mover_symbols)].copy()
                    results = results.sort_values('probability', ascending=False)
                    results['rank'] = range(1, len(results) + 1)

        # Save to database
        if save_to_db:
            self._save_predictions(results)

        # Return top K
        top_picks = results.head(top_k).copy()

        print(f"\nTop {top_k} Predictions:")
        print("-" * 50)
        value_label = "Score" if self.model_type == "ranking" else "Probability"
        for _, row in top_picks.iterrows():
            print(f"  {row['rank']:2d}. {row['symbol']:6s} - {value_label}: {row['probability']:.4f}")

        return top_picks

    def _get_mover_symbols(self, date, mover_types: List[str]) -> List[str]:
        """Get symbols that are in movers list for a given date"""
        from database import session_scope, Stock, DailyMover

        with session_scope() as session:
            rows = session.query(Stock.symbol).join(DailyMover, DailyMover.stock_id == Stock.id).filter(
                DailyMover.date == date,
                DailyMover.mover_type.in_(mover_types)
            ).all()

        return [r[0] for r in rows]

    def _save_predictions(self, results: pd.DataFrame) -> None:
        """Save predictions to database"""
        from utils.holidays import next_trading_day

        prediction_date = datetime.now().date()
        # Use proper holiday calendar to get next trading day
        target_date = next_trading_day(prediction_date)

        with session_scope() as session:
            for _, row in results.iterrows():
                stock = session.query(Stock).filter(Stock.symbol == row['symbol']).first()
                if not stock:
                    continue

                # Check if prediction exists
                existing = session.query(Prediction).filter(
                    Prediction.stock_id == stock.id,
                    Prediction.prediction_date == prediction_date
                ).first()

                if existing:
                    existing.probability = row['probability']
                    existing.rank = row['rank']
                    existing.model_version = self.model_name
                else:
                    pred = Prediction(
                        stock_id=stock.id,
                        prediction_date=prediction_date,
                        target_date=target_date,
                        probability=row['probability'],
                        rank=row['rank'],
                        model_version=self.model_name,
                    )
                    session.add(pred)

    def get_prediction_history(
        self,
        start_date: str = None,
        end_date: str = None,
        top_k: int = None
    ) -> pd.DataFrame:
        """
        Get historical predictions and their outcomes

        Args:
            start_date: Start date
            end_date: End date
            top_k: Only include top K predictions per day

        Returns:
            DataFrame with prediction history
        """
        top_k = top_k or TOP_PICKS_COUNT

        with session_scope() as session:
            query = session.query(Prediction).join(Stock)

            if start_date:
                query = query.filter(Prediction.prediction_date >= start_date)
            if end_date:
                query = query.filter(Prediction.prediction_date <= end_date)

            # Filter top K per day
            query = query.filter(Prediction.rank <= top_k)
            query = query.order_by(Prediction.prediction_date.desc(), Prediction.rank)

            results = []
            for pred in query.all():
                results.append({
                    'prediction_date': pred.prediction_date,
                    'target_date': pred.target_date,
                    'symbol': pred.stock.symbol,
                    'probability': pred.probability,
                    'rank': pred.rank,
                    'actual_return': pred.actual_return,
                    'is_top_gainer': pred.is_top_gainer,
                    'is_correct': pred.is_correct,
                })

            return pd.DataFrame(results)

    def update_actuals(self, date: str = None) -> int:
        """
        Update predictions with actual returns

        Args:
            date: Target date to update (default: yesterday)

        Returns:
            Number of predictions updated
        """
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date()
        else:
            date = datetime.strptime(date, "%Y-%m-%d").date()

        updated = 0

        with session_scope() as session:
            # Get predictions for this target date
            predictions = session.query(Prediction).filter(
                Prediction.target_date == date,
                Prediction.actual_return == None
            ).all()

            for pred in predictions:
                # Get actual price data
                price_data = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == pred.stock_id,
                    DailyPrice.date == date
                ).first()

                if price_data and price_data.close and price_data.close != 0:
                    prior_price = session.query(DailyPrice).filter(
                        DailyPrice.stock_id == pred.stock_id,
                        DailyPrice.date < date
                    ).order_by(DailyPrice.date.desc()).first()
                    if not prior_price or not prior_price.close:
                        continue

                    pred.actual_return = (price_data.close - prior_price.close) / prior_price.close
                    pred.is_top_gainer = (pred.actual_return >= TOP_GAINER_THRESHOLD)

                    if self.model_type == "ranking":
                        pred.is_correct = (pred.rank is not None and pred.rank <= TOP_PICKS_COUNT) and pred.is_top_gainer
                    else:
                        pred.is_correct = (pred.probability >= 0.5) == pred.is_top_gainer
                    updated += 1

        print(f"Updated {updated} predictions with actual returns")
        return updated


def run_prediction():
    """CLI function to run predictions"""
    import argparse

    parser = argparse.ArgumentParser(description="Run top gainer predictions")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--top", type=int, default=TOP_PICKS_COUNT, help="Number of top picks")
    parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    args = parser.parse_args()

    predictor = Predictor(model_name=args.model)
    results = predictor.predict(top_k=args.top, save_to_db=not args.no_save)

    # Print results
    if not results.empty:
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        print(results.to_string(index=False))


if __name__ == "__main__":
    run_prediction()
