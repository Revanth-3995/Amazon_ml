"""
Submission Adapter Module.
Formats predictions into submission files and marks official format status.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from config.default_config import Config
from utils.memory import log_disk_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("11_build_submission")


def build_submission(config: Config) -> None:
    """Build submission file from entity predictions."""
    config.ensure_directories()

    pred_csv = config.predictions_dir / "entity_matches.csv"
    sub_csv = config.predictions_dir / "submission.csv"

    logger.warning("OFFICIAL SUBMISSION FORMAT NOT YET VERIFIED. Generating standard internal format adapter.")

    if not pred_csv.exists():
        logger.warning(f"Prediction file {pred_csv} not found. Creating empty submission template.")
        df_sub = pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])
    else:
        df_sub = pd.read_csv(pred_csv)

    df_sub.to_csv(sub_csv, index=False)
    logger.info(f"Saved submission adapter output to {sub_csv} ({len(df_sub)} rows)")

    log_disk_usage(config.predictions_dir, "Predictions Directory (Submission)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ER Submission")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    build_submission(cfg)
