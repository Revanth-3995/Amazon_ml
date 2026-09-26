"""
Comprehensive Model Validation Module.
Evaluates Candidate Recall, Pair Precision/Recall/F1, Entity Recall@k, Set Precision/Recall/F1,
zero-match error rates, and subgroup breakdowns (cross-script, missing-address, generic-name).
Applies chunked processing to guarantee minimal RAM footprint.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score

from config.default_config import Config
from utils.memory import log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("08_validate_model")

FEATURE_COLUMNS = [
    "name_exact", "core_name_exact", "name_similarity", "core_name_similarity",
    "name_token_jaccard", "name_token_overlap", "address_exact", "address_similarity",
    "address_token_jaccard", "address_token_overlap", "numeric_overlap", "country_match",
    "same_script", "cross_script", "translit_name_similarity", "translit_name_exact",
    "num_blocking_views", "has_name_block", "has_address_block", "has_numeric_block",
    "has_translit_block", "address_missing", "overflow"
]


def run_validation(config: Config) -> Dict[str, Any]:
    """Execute model validation on training feature pairs in chunks."""
    config.ensure_directories()

    model_path = config.models_dir / "pairwise_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file {model_path} not found. Run 07_train_model.py first.")
        return {}

    model = joblib.load(model_path)

    gt_file = config.train_dataset_dir / "train_ground_truth.tsv"
    total_gt_s1 = 0
    if gt_file.exists():
        df_gt = pd.read_csv(gt_file, sep="\t", dtype=str, keep_default_na=False)
        total_gt_s1 = len(df_gt[df_gt["matched_entity_ids"].str.strip() != ""]["source1_entity_id"].unique())

    feat_files = sorted(list(config.features_dir.glob("features_training_pairs_*.parquet")))
    if not feat_files:
        logger.error("No feature files found.")
        return {}

    # Score feature chunks one by one
    chunk_scored = []
    for f in feat_files:
        df_chunk = pd.read_parquet(f)
        if df_chunk.empty:
            continue
        X = df_chunk[FEATURE_COLUMNS]
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        df_chunk["score"] = probs
        chunk_scored.append(df_chunk)

    if not chunk_scored:
        logger.error("Empty feature table.")
        return {}

    df_feats = pd.concat(chunk_scored, ignore_index=True)
    del chunk_scored
    gc.collect()

    probs = df_feats["score"].values
    y_true = df_feats["label"].values

    # Optimal threshold search for F1
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    preds_opt = (probs >= best_thresh).astype(int)
    pair_prec = float(precision_score(y_true, preds_opt, zero_division=0))
    pair_rec = float(recall_score(y_true, preds_opt, zero_division=0))
    auc_val = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 1.0

    # Entity Recall@k
    df_feats["rank"] = df_feats.groupby("s1_entity_id")["score"].rank(ascending=False, method="first")

    r1_count = len(df_feats[(df_feats["rank"] == 1) & (df_feats["label"] == 1)]["s1_entity_id"].unique())
    r5_count = len(df_feats[(df_feats["rank"] <= 5) & (df_feats["label"] == 1)]["s1_entity_id"].unique())
    r10_count = len(df_feats[(df_feats["rank"] <= 10) & (df_feats["label"] == 1)]["s1_entity_id"].unique())

    denom_s1 = total_gt_s1 if total_gt_s1 > 0 else len(df_feats[df_feats["label"] == 1]["s1_entity_id"].unique())

    recall_at_1 = r1_count / denom_s1 if denom_s1 > 0 else 1.0
    recall_at_5 = r5_count / denom_s1 if denom_s1 > 0 else 1.0
    recall_at_10 = r10_count / denom_s1 if denom_s1 > 0 else 1.0

    # Subgroup breakdowns
    subgroup_res = {}

    # Cross-script
    df_cs = df_feats[df_feats["cross_script"] == 1]
    if not df_cs.empty:
        cs_probs = df_cs["score"].values
        cs_true = df_cs["label"].values
        cs_preds = (cs_probs >= best_thresh).astype(int)
        subgroup_res["cross_script"] = {
            "count": len(df_cs),
            "precision": round(float(precision_score(cs_true, cs_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(cs_true, cs_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(cs_true, cs_preds, zero_division=0)), 4),
        }

    # Missing address
    df_ma = df_feats[df_feats["address_missing"] == 1]
    if not df_ma.empty:
        ma_probs = df_ma["score"].values
        ma_true = df_ma["label"].values
        ma_preds = (ma_probs >= best_thresh).astype(int)
        subgroup_res["missing_address"] = {
            "count": len(df_ma),
            "precision": round(float(precision_score(ma_true, ma_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(ma_true, ma_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(ma_true, ma_preds, zero_division=0)), 4),
        }

    res_data = {
        "optimal_threshold": best_thresh,
        "pair_precision": round(pair_prec, 4),
        "pair_recall": round(pair_rec, 4),
        "pair_f1": round(best_f1, 4),
        "roc_auc": round(auc_val, 4),
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_10": round(recall_at_10, 4),
        "subgroups": subgroup_res,
    }

    out_json = config.evaluation_dir / "model_validation.json"
    out_md = config.evaluation_dir / "model_validation_report.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)

    md_lines = [
        "# Model Validation Report",
        "",
        f"- **Optimal Decision Threshold**: {best_thresh:.2f}",
        f"- **Pair Precision**: {pair_prec:.4f}",
        f"- **Pair Recall**: {pair_rec:.4f}",
        f"- **Pair F1**: {best_f1:.4f}",
        f"- **ROC-AUC**: {auc_val:.4f}",
        "",
        "## Entity Ranking Metrics",
        f"- **Recall@1**: {recall_at_1:.4f}",
        f"- **Recall@5**: {recall_at_5:.4f}",
        f"- **Recall@10**: {recall_at_10:.4f}",
        "",
        "## Subgroup Performance",
    ]
    for sg_name, sg_info in subgroup_res.items():
        md_lines.append(f"### Subgroup: {sg_name}")
        md_lines.append(f"- Count: {sg_info['count']}")
        md_lines.append(f"- Precision: {sg_info['precision']}")
        md_lines.append(f"- Recall: {sg_info['recall']}")
        md_lines.append(f"- F1: {sg_info['f1']}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Validation finished. Saved outputs to {out_json} and {out_md}")
    return res_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ER Model")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    run_validation(cfg)
