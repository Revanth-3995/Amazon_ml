"""
Inference & Entity Ranking Module.
Predicts match probabilities on test candidate chunks incrementally, applies decision thresholds,
allows zero / one-to-many match set decisions, and streams prediction outputs safely.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import joblib
import pandas as pd

from config.default_config import Config
from utils.memory import log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("10_predict")

FEATURE_COLUMNS = [
    "name_exact", "core_name_exact", "name_similarity", "core_name_similarity",
    "name_token_jaccard", "name_token_overlap", "address_exact", "address_similarity",
    "address_token_jaccard", "address_token_overlap", "numeric_overlap", "country_match",
    "same_script", "cross_script", "translit_name_similarity", "translit_name_exact",
    "num_blocking_views", "has_name_block", "has_address_block", "has_numeric_block",
    "has_translit_block", "address_missing", "overflow"
]


def run_prediction(config: Config, threshold: float = None) -> None:
    """Predict match probabilities on test feature chunks and apply entity match decisions."""
    config.ensure_directories()

    model_path = config.models_dir / "pairwise_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file {model_path} not found. Run 07_train_model.py first.")
        return

    model = joblib.load(model_path)

    if threshold is None:
        val_json = config.evaluation_dir / "model_validation.json"
        if val_json.exists():
            with open(val_json, "r", encoding="utf-8") as f:
                val_data = json.load(f)
                threshold = val_data.get("optimal_threshold", 0.5)
        else:
            threshold = 0.5

    logger.info(f"Using match decision threshold: {threshold:.2f}")

    test_feat_files = sorted(list(config.features_dir.glob("features_candidates_test_*.parquet")))
    if not test_feat_files:
        logger.warning(f"No test feature files found in {config.features_dir}. Run 06_build_features.py --is-test first.")
        return

    matched_pairs_list = []
    raw_pred_files = []

    for feat_file in test_feat_files:
        df_feat = pd.read_parquet(feat_file)
        if df_feat.empty:
            continue

        X = df_feat[FEATURE_COLUMNS]
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        df_feat["match_score"] = probs
        df_feat["is_match"] = (probs >= threshold).astype(int)

        out_chunk_parquet = config.predictions_dir / f"pred_{feat_file.name}"
        df_feat[["s1_entity_id", "target_entity_id", "match_score", "is_match"]].to_parquet(out_chunk_parquet, index=False)
        raw_pred_files.append(out_chunk_parquet)

        # Collect positive matches for entity aggregation
        df_matches = df_feat[df_feat["is_match"] == 1][["s1_entity_id", "target_entity_id"]]
        if not df_matches.empty:
            matched_pairs_list.append(df_matches)

        del df_feat
        gc.collect()

    out_parquet = config.predictions_dir / "raw_predictions.parquet"
    out_csv = config.predictions_dir / "entity_matches.csv"

    # Merge raw predictions into single Parquet file if non-empty
    if raw_pred_files:
        df_all_raw = pd.concat([pd.read_parquet(f) for f in raw_pred_files], ignore_index=True)
        df_all_raw.to_parquet(out_parquet, index=False)
    else:
        pd.DataFrame(columns=["s1_entity_id", "target_entity_id", "match_score", "is_match"]).to_parquet(out_parquet, index=False)

    if matched_pairs_list:
        df_all_matches = pd.concat(matched_pairs_list, ignore_index=True)
        grouped = df_all_matches.groupby("s1_entity_id")["target_entity_id"].apply(lambda ids: " ".join(sorted(set(ids)))).reset_index()
        grouped.rename(columns={"s1_entity_id": "source1_entity_id", "target_entity_id": "matched_entity_ids"}, inplace=True)
        grouped.to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"]).to_csv(out_csv, index=False)

    logger.info(f"Saved prediction outputs: {out_parquet} and {out_csv}")

    log_memory("Prediction Complete")
    log_disk_usage(config.predictions_dir, "Predictions Directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Test Entity Matches")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold for matching")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    run_prediction(cfg, threshold=args.threshold)
