"""
Google Colab Pipeline Runner and Orchestrator.
Supports step-by-step or full execution with chunk size overrides, memory safety limits, and stage resumability.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

from config.default_config import Config
from utils.memory import get_rss_mb, log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline_colab")


def run_stage(stage_name: str, config: Config, resume: bool = True) -> None:
    """Dynamically import and run a specific pipeline stage."""
    logger.info(f"========== Starting Pipeline Stage: {stage_name} ==========")
    log_memory(f"Before {stage_name}")

    try:
        if stage_name == "preprocess":
            mod = importlib.import_module("code.01_preprocess")
            mod.run_preprocessing(config, resume=resume)

        elif stage_name == "indexes":
            mod = importlib.import_module("code.02_build_indexes")
            mod.build_indexes(config, resume=resume)

        elif stage_name == "benchmark":
            mod = importlib.import_module("code.03_blocking_benchmark")
            mod.run_blocking_benchmark(config)

        elif stage_name == "candidates":
            mod = importlib.import_module("code.04_generate_candidates")
            mod.generate_candidates(config, resume=resume)

        elif stage_name == "pairs":
            mod = importlib.import_module("code.05_build_training_pairs")
            mod.build_training_pairs(config, resume=resume)

        elif stage_name == "features":
            mod = importlib.import_module("code.06_build_features")
            mod.build_features(config, is_test=False, resume=resume)

        elif stage_name == "train":
            mod = importlib.import_module("code.07_train_model")
            mod.train_pairwise_model(config)

        elif stage_name == "validate":
            mod = importlib.import_module("code.08_validate_model")
            mod.run_validation(config)

        elif stage_name == "test_candidates":
            mod = importlib.import_module("code.09_generate_test_candidates")
            mod.generate_test_candidates(config, resume=resume)
            # Also build features for test candidates
            mod_feat = importlib.import_module("code.06_build_features")
            mod_feat.build_features(config, is_test=True, resume=resume)

        elif stage_name == "predict":
            mod = importlib.import_module("code.10_predict")
            mod.run_prediction(config)

        elif stage_name == "submission":
            mod = importlib.import_module("code.11_build_submission")
            mod.build_submission(config)

        else:
            logger.error(f"Unknown stage name: {stage_name}")

    except Exception as e:
        logger.error(f"Error executing stage {stage_name}: {e}", exc_info=True)
        log_memory(f"Error {stage_name}")
        sys.exit(1)

    log_memory(f"After {stage_name}")
    logger.info(f"========== Completed Pipeline Stage: {stage_name} ==========\n")


def main():
    parser = argparse.ArgumentParser(description="Orchestrate Colab Business Entity Resolution Pipeline")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path in Google Drive or local")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Initial default chunk size (default: 50,000)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all", "preprocess", "indexes", "benchmark", "candidates",
            "pairs", "features", "train", "validate", "test_candidates",
            "predict", "submission"
        ],
        help="Stage to execute (or 'all' for complete pipeline)"
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume and restart stage")
    parser.add_argument("--max-memory-mb", type=float, default=4000.0, help="Memory warning threshold in MB")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root, chunk_size=args.chunk_size)
    cfg.memory_warning_mb = args.max_memory_mb
    cfg.ensure_directories()

    resume = not args.no_resume

    stages_order = [
        "preprocess", "indexes", "benchmark", "candidates",
        "pairs", "features", "train", "validate", "test_candidates",
        "predict", "submission"
    ]

    if args.stage == "all":
        for s in stages_order:
            run_stage(s, cfg, resume=resume)
    else:
        run_stage(args.stage, cfg, resume=resume)


if __name__ == "__main__":
    main()
