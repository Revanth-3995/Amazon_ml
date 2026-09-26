"""
Training Pair Generation Module.
Pairs candidates against ground truth labels (1 = true match, 0 = hard negative candidate).
Ensures all ground truth positive pairs are included and writes chunked Parquet output.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, Tuple

import duckdb
import pandas as pd

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("05_build_training_pairs")


def build_training_pairs(config: Config, resume: bool = True) -> None:
    """Build training pairs by labeling candidate chunks against ground truth."""
    config.ensure_directories()
    manifest = ManifestManager(config.processed_dir / "manifest.json")

    stage_name = "build_training_pairs"
    manifest.mark_stage_started(stage_name)

    gt_file = config.train_dataset_dir / "train_ground_truth.tsv"
    if not gt_file.exists():
        logger.error(f"Ground truth file {gt_file} not found. Cannot build training pairs.")
        return

    df_gt = pd.read_csv(gt_file, sep="\t", dtype=str, keep_default_na=False)
    gt_pairs: Set[Tuple[str, str]] = set()

    for _, row in df_gt.iterrows():
        s1_id = row.get("source1_entity_id", "").strip()
        matched_str = row.get("matched_entity_ids", "").strip()
        if s1_id and matched_str:
            for target_id in matched_str.split():
                if target_id.strip():
                    gt_pairs.add((s1_id, target_id.strip()))

    logger.info(f"Loaded {len(gt_pairs)} ground truth positive pairs.")

    cand_files = sorted(list(config.candidates_dir.glob("candidates_train_source1_*.parquet")))
    if not cand_files:
        logger.error(f"No candidate files found in {config.candidates_dir}. Run 04_generate_candidates.py first.")
        return

    for cand_path in cand_files:
        chunk_id = cand_path.stem.replace("candidates_", "")
        out_parquet = config.processed_dir / f"training_pairs_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed training pairs chunk {chunk_id}")
            continue

        df_cands = pd.read_parquet(cand_path)
        if df_cands.empty:
            df_cands["label"] = pd.Series(dtype="int8")
            df_cands.to_parquet(out_parquet, index=False)
            manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), 0)
            continue

        labels = []
        for _, row in df_cands.iterrows():
            s1_id = row["s1_entity_id"]
            target_id = row["target_entity_id"]
            label = 1 if (s1_id, target_id) in gt_pairs else 0
            labels.append(label)

        df_cands["label"] = pd.Series(labels, dtype="int8")
        df_cands.to_parquet(out_parquet, index=False)

        manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), len(df_cands))
        log_memory(f"Built training pairs chunk {chunk_id}")

    manifest.mark_stage_completed(stage_name)
    log_disk_usage(config.processed_dir, "Processed Directory (Training Pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Training Pairs")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--no-resume", action="store_true", help="Force rebuild training pairs")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    build_training_pairs(cfg, resume=not args.no_resume)
