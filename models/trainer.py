"""
Model Trainer - Train LightGBM model for top gainer prediction
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from config import TEST_SIZE, MODELS_DIR, TOP_PICKS_COUNT
from features.pipeline import FeaturePipeline


class ModelTrainer:
    """Train and evaluate LightGBM model"""

    def __init__(self, model_name: str = None):
        self.model = None
        self.feature_names = None
        self.model_name = model_name or f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {}
        self.feature_importance = None

    def get_default_params(self) -> Dict:
        """Get default LightGBM parameters"""
        return {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,  # Handle class imbalance
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
        }

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None,
        use_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training

        Args:
            X: Features DataFrame
            y: Labels Series
            test_size: Test split ratio
            use_smote: Use SMOTE for oversampling

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or TEST_SIZE

        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        self.feature_names = X_numeric.columns.tolist()

        # Replace inf and fill NaN
        X_clean = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())

        # Convert to numpy
        X_array = X_clean.values
        y_array = y.values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array,
            test_size=test_size,
            random_state=42,
            stratify=y_array
        )

        # Apply SMOTE for class imbalance
        if use_smote and y_train.sum() > 5:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"SMOTE applied: {len(y_train)} training samples")
            except Exception as e:
                print(f"SMOTE failed, using original data: {e}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict = None,
        use_smote: bool = True,
        test_size: float = None
    ) -> Dict:
        """
        Train LightGBM model

        Args:
            X: Features DataFrame
            y: Labels Series
            params: LightGBM parameters
            use_smote: Use SMOTE oversampling
            test_size: Test split ratio

        Returns:
            Dict with training metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, test_size=test_size, use_smote=use_smote
        )

        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        print(f"Positive class in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

        # Get parameters
        params = params or self.get_default_params()

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, feature_name=self.feature_names, reference=train_data)

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            num_boost_round=params.get('n_estimators', 500),
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        self.metrics = self.evaluate(X_test, y_test)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        return self.metrics

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate model performance

        Args:
            X: Test features
            y: Test labels
            threshold: Classification threshold

        Returns:
            Dict with evaluation metrics
        """
        # Get probabilities
        y_proba = self.model.predict(X)
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y).mean(),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if y.sum() > 0 else 0,
            'avg_precision': average_precision_score(y, y_proba) if y.sum() > 0 else 0,
        }

        # Precision at K (most important metric)
        k = TOP_PICKS_COUNT
        top_k_idx = np.argsort(y_proba)[-k:]
        precision_at_k = y[top_k_idx].sum() / k
        metrics['precision_at_k'] = precision_at_k
        metrics['k'] = k

        # Print report
        print("\n" + "=" * 50)
        print("Model Evaluation Results")
        print("=" * 50)
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1 Score:       {metrics['f1']:.4f}")
        print(f"ROC AUC:        {metrics['roc_auc']:.4f}")
        print(f"Avg Precision:  {metrics['avg_precision']:.4f}")
        print(f"Precision@{k}:   {precision_at_k:.4f}")
        print("=" * 50)

        return metrics


    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        params: Dict = None
    ) -> Dict:
        """
        Time series cross-validation

        Args:
            X: Features DataFrame
            y: Labels Series
            n_splits: Number of CV splits
            params: LightGBM parameters

        Returns:
            Dict with CV metrics
        """
        X_numeric = X.select_dtypes(include=[np.number])
        X_clean = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(X_numeric.median())

        params = params or self.get_default_params()
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_clean)):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create model for this fold
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[test_data],
                num_boost_round=200,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # Evaluate
            y_proba = model.predict(X_test)
            y_pred = (y_proba >= 0.5).astype(int)

            fold_metrics = {
                'fold': fold + 1,
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }

            # Precision@K
            k = min(TOP_PICKS_COUNT, len(y_test))
            if k > 0:
                top_k_idx = np.argsort(y_proba)[-k:]
                fold_metrics['precision_at_k'] = y_test.iloc[top_k_idx].sum() / k

            cv_metrics.append(fold_metrics)
            print(f"Fold {fold + 1}: Precision@{k} = {fold_metrics.get('precision_at_k', 0):.4f}")

        # Aggregate metrics
        cv_df = pd.DataFrame(cv_metrics)
        summary = {
            'mean_precision': cv_df['precision'].mean(),
            'mean_recall': cv_df['recall'].mean(),
            'mean_f1': cv_df['f1'].mean(),
            'mean_precision_at_k': cv_df['precision_at_k'].mean(),
            'std_precision_at_k': cv_df['precision_at_k'].std(),
        }

        print(f"\nCV Summary:")
        print(f"Mean Precision@{TOP_PICKS_COUNT}: {summary['mean_precision_at_k']:.4f} "
              f"(+/- {summary['std_precision_at_k']:.4f})")

        return summary

    def save(self, path: str = None) -> str:
        """
        Save model and metadata

        Args:
            path: Save directory (default: MODELS_DIR)

        Returns:
            Path to saved model
        """
        path = path or MODELS_DIR
        os.makedirs(path, exist_ok=True)

        # Save model
        model_path = os.path.join(path, f"{self.model_name}.txt")
        self.model.save_model(model_path)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'metrics': self.metrics,
        }
        metadata_path = os.path.join(path, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save feature importance
        if self.feature_importance is not None:
            fi_path = os.path.join(path, f"{self.model_name}_feature_importance.csv")
            self.feature_importance.to_csv(fi_path, index=False)


        print(f"\nModel saved to: {model_path}")
        return model_path

    def load(self, model_name: str, path: str = None) -> None:
        """
        Load model and metadata

        Args:
            model_name: Model name
            path: Model directory
        """
        path = path or MODELS_DIR

        # Load model
        model_path = os.path.join(path, f"{model_name}.txt")
        self.model = lgb.Booster(model_file=model_path)

        # Load metadata
        metadata_path = os.path.join(path, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_name = metadata.get('model_name', model_name)
            self.feature_names = metadata.get('feature_names', [])
            self.metrics = metadata.get('metrics', {})

        print(f"Model loaded from: {model_path}")

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(n)


def train_model():
    """CLI function to train model"""
    import argparse

    parser = argparse.ArgumentParser(description="Train top gainer prediction model")
    parser.add_argument("--start", type=str, help="Start date for training data")
    parser.add_argument("--end", type=str, help="End date for training data")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    args = parser.parse_args()

    # Build dataset
    print("Building training dataset...")
    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset(start_date=args.start, end_date=args.end)

    if X.empty:
        print("No data available for training!")
        return

    # Prepare features
    X_train, y_train, feature_names = pipeline.prepare_for_training(X, y)

    # Train model
    print("\nTraining model...")
    trainer = ModelTrainer(model_name=args.name)
    trainer.train(X_train, y_train, use_smote=not args.no_smote)

    # Save model
    trainer.save()

    # Print top features
    print("\nTop 20 Important Features:")
    print(trainer.get_top_features(20))


if __name__ == "__main__":
    train_model()
