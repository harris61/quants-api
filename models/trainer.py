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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
from config import TEST_SIZE, MODELS_DIR, TOP_PICKS_COUNT
from features.pipeline import FeaturePipeline


class ModelTrainer:
    """Train and evaluate LightGBM model"""

    def __init__(self, model_name: str = None):
        self.model = None
        self.feature_names = None
        self.feature_medians = None
        self.model_type = "classification"
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
        self.feature_medians = X_clean.median().to_dict()
        X_clean = X_clean.fillna(X_clean.median())

        # Temporal split (no shuffling). Use positional alignment to avoid
        # index-duplication expanding labels when dates repeat across stocks.
        y_aligned = pd.Series(y.to_numpy(), index=X_clean.index)
        _, indexer = X_clean.index.sort_values(return_indexer=True)
        X_sorted = X_clean.iloc[indexer]
        y_sorted = y_aligned.iloc[indexer]

        split_idx = int(len(X_sorted) * (1 - test_size))
        X_train = X_sorted.iloc[:split_idx].values
        X_test = X_sorted.iloc[split_idx:].values
        y_train = y_sorted.iloc[:split_idx].values
        y_test = y_sorted.iloc[split_idx:].values

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

        # Class weighting (time-series safe alternative to SMOTE)
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        params['scale_pos_weight'] = scale_pos_weight
        params.pop('is_unbalance', None)

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
            'model_type': self.model_type,
            'created_at': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians,
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
            self.model_type = metadata.get('model_type', 'classification')
            self.feature_names = metadata.get('feature_names', [])
            self.feature_medians = metadata.get('feature_medians', {})
            self.metrics = metadata.get('metrics', {})

        print(f"Model loaded from: {model_path}")

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(n)


class RankingTrainer:
    """Train and evaluate LightGBM LambdaRank model"""

    def __init__(self, model_name: str = None):
        self.model = None
        self.feature_names = None
        self.feature_medians = None
        self.model_type = "ranking"
        self.model_name = model_name or f"lgbm_rank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {}
        self.feature_importance = None

    def get_default_params(self) -> Dict:
        return {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'max_depth': -1,
            'n_estimators': 500,
            'early_stopping_rounds': 50,
        }

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare ranking data with date-based groups.

        Returns:
            X_train, X_test, y_train, y_test, groups_train, groups_test
        """
        test_size = test_size or TEST_SIZE

        X_numeric = X.select_dtypes(include=[np.number])
        self.feature_names = X_numeric.columns.tolist()

        X_clean = X_numeric.replace([np.inf, -np.inf], np.nan)
        self.feature_medians = X_clean.median().to_dict()
        X_clean = X_clean.fillna(X_clean.median())

        # Align by position to avoid duplicate-index expansion
        y_aligned = pd.Series(y.to_numpy(), index=X_clean.index)
        data = X_clean.copy()
        data["_date"] = pd.to_datetime(data.index).normalize()
        data["label"] = y_aligned
        data = data.dropna(subset=["label"])

        unique_dates = sorted(data["_date"].unique())
        split_idx = int(len(unique_dates) * (1 - test_size))
        train_dates = set(unique_dates[:split_idx])
        test_dates = set(unique_dates[split_idx:])

        train_df = data[data["_date"].isin(train_dates)]
        test_df = data[data["_date"].isin(test_dates)]

        groups_train = train_df.groupby("_date").size().values
        groups_test = test_df.groupby("_date").size().values

        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        return X_train, X_test, y_train, y_test, groups_train, groups_test

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict = None,
        test_size: float = None
    ) -> Dict:
        X_train, X_test, y_train, y_test, groups_train, groups_test = self.prepare_data(
            X, y, test_size=test_size
        )

        print(f"Training data: {len(X_train)} samples in {len(groups_train)} groups")
        print(f"Test data: {len(X_test)} samples in {len(groups_test)} groups")

        params = params or self.get_default_params()

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            feature_name=self.feature_names
        )
        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            group=groups_test,
            feature_name=self.feature_names,
            reference=train_data
        )

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

        self.metrics = self.evaluate(X_test, y_test, groups_test)

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        return self.metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = None) -> Dict:
        k = k or TOP_PICKS_COUNT
        scores = self.model.predict(X)

        ndcgs = []
        idx = 0
        for group_size in groups:
            group_scores = scores[idx:idx + group_size]
            group_labels = y[idx:idx + group_size]
            idx += group_size

            order = np.argsort(group_scores)[::-1]
            gains = group_labels[order][:k]
            discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
            dcg = np.sum(gains * discounts)

            ideal = np.sort(group_labels)[::-1][:k]
            ideal_dcg = np.sum(ideal * discounts) if len(ideal) else 0
            ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0)

        metrics = {
            'ndcg_at_k': float(np.mean(ndcgs)) if ndcgs else 0.0,
            'k': k
        }

        print("\n" + "=" * 50)
        print("Ranking Evaluation Results")
        print("=" * 50)
        print(f"NDCG@{k}:        {metrics['ndcg_at_k']:.4f}")
        print("=" * 50)

        return metrics

    def save(self, path: str = None) -> str:
        path = path or MODELS_DIR
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, f"{self.model_name}.txt")
        self.model.save_model(model_path)

        metadata = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'created_at': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians,
            'metrics': self.metrics,
        }
        metadata_path = os.path.join(path, f"{self.model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.feature_importance is not None:
            fi_path = os.path.join(path, f"{self.model_name}_feature_importance.csv")
            self.feature_importance.to_csv(fi_path, index=False)

        print(f"\nModel saved to: {model_path}")
        return model_path

    def load(self, model_name: str, path: str = None) -> None:
        path = path or MODELS_DIR

        model_path = os.path.join(path, f"{model_name}.txt")
        self.model = lgb.Booster(model_file=model_path)

        metadata_path = os.path.join(path, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_name = metadata.get('model_name', model_name)
            self.model_type = metadata.get('model_type', 'ranking')
            self.feature_names = metadata.get('feature_names', [])
            self.feature_medians = metadata.get('feature_medians', {})
            self.metrics = metadata.get('metrics', {})

        print(f"Model loaded from: {model_path}")

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            return pd.DataFrame()
        return self.feature_importance.head(n)


def train_model():
    """CLI function to train model"""
    import argparse
    from config import INCLUDE_DELISTED_IN_TRAINING

    parser = argparse.ArgumentParser(description="Train top gainer prediction model")
    parser.add_argument("--start", type=str, help="Start date for training data")
    parser.add_argument("--end", type=str, help="End date for training data")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--ranking", action="store_true", help="Train a LambdaRank model")
    parser.add_argument("--no-delisted", action="store_true",
                        help="Exclude delisted stocks (not recommended, causes survivorship bias)")
    args = parser.parse_args()

    # Build dataset
    include_delisted = INCLUDE_DELISTED_IN_TRAINING and not args.no_delisted
    print(f"Building training dataset (include_delisted={include_delisted})...")
    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset(
        start_date=args.start,
        end_date=args.end,
        label_type="return" if args.ranking else "binary",
        include_delisted=include_delisted
    )

    if X.empty:
        print("No data available for training!")
        return

    # Prepare features
    X_train, y_train, feature_names = pipeline.prepare_for_training(X, y)

    # Train model
    print("\nTraining model...")
    if args.ranking:
        trainer = RankingTrainer(model_name=args.name)
        trainer.train(X_train, y_train)
    else:
        trainer = ModelTrainer(model_name=args.name)
        trainer.train(X_train, y_train, use_smote=not args.no_smote)

    # Save model
    trainer.save()

    # Print top features
    print("\nTop 20 Important Features:")
    print(trainer.get_top_features(20))


if __name__ == "__main__":
    train_model()
