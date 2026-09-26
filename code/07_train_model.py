"""
Pairwise Model Training Module.
Trains XGBoost / LightGBM pairwise score models using entity-level grouped splits (preventing entity leakage).
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config.default_config import Config
from utils.memory import log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("07_train_model")

FEATURE_COLUMNS = [
    "name_exact", "core_name_exact", "name_similarity", "core_name_similarity",
    "name_token_jaccard", "name_token_overlap", "address_exact", "address_similarity",
    "address_token_jaccard", "address_token_overlap", "numeric_overlap", "country_match",
    "same_script", "cross_script", "translit_name_similarity", "translit_name_exact",
    "num_blocking_views", "has_name_block", "has_address_block", "has_numeric_block",
    "has_translit_block", "address_missing", "overflow"
]


def train_pairwise_model(config: Config) -> Dict[str, Any]:
    """Train XGBoost or LightGBM model on feature chunks using GroupShuffleSplit."""
    config.ensure_directories()

    feat_files = sorted(list(config.features_dir.glob("features_training_pairs_*.parquet")))
    if not feat_files:
        logger.error(f"No feature files found in {config.features_dir}. Run 06_build_features.py first.")
        return {}

    # Read feature chunks
    df_list = [pd.read_parquet(f) for f in feat_files]
    df_all = pd.concat(df_list, ignore_index=True)
    if df_all.empty:
        logger.error("Feature dataset is empty.")
        return {}

    logger.info(f"Loaded feature dataset: {len(df_all)} pairs.")

    X = df_all[FEATURE_COLUMNS]
    y = df_all["label"].values
    groups = df_all["s1_internal_id"].values

    # Entity-level grouped split (train 80%, val 20%)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    logger.info(f"Grouped Split -> Train: {len(X_train)} pairs | Val: {len(X_val)} pairs.")

    model = None
    if HAS_XGB:
        logger.info("Training XGBoost model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=config.seed,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif HAS_LGB:
        logger.info("Training LightGBM model...")
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=config.seed,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(5, verbose=False)])
    else:
        from sklearn.ensemble import RandomForestClassifier
        logger.info("Training RandomForestClassifier fallback model...")
        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=config.seed, n_jobs=2)
        model.fit(X_train, y_train)

    # Evaluate validation AUC
    val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
    val_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 1.0

    precision, recall, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = float(auc(recall, precision))

    logger.info(f"Validation ROC-AUC: {val_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Save model
    model_path = config.models_dir / "pairwise_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    log_memory("Train Model Complete")
    log_disk_usage(config.models_dir, "Models Directory")

    return {
        "val_roc_auc": round(val_auc, 4),
        "val_pr_auc": round(pr_auc, 4),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pairwise ER Model")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    train_pairwise_model(cfg)
