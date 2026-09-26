"""
End-to-end integration test executing the complete pipeline via run_pipeline_colab.py on synthetic data.
"""

import importlib
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from config.default_config import Config
from tests.fixtures.synthetic_data import generate_synthetic_dataset

run_colab_mod = importlib.import_module("code.run_pipeline_colab")
run_stage = run_colab_mod.run_stage


def test_end_to_end_pipeline_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "dataset"
        generate_synthetic_dataset(dataset_dir)

        cfg = Config(project_root=str(tmp_path), chunk_size=15)
        cfg.ensure_directories()

        stages = [
            "preprocess", "indexes", "benchmark", "candidates",
            "pairs", "features", "train", "validate", "test_candidates",
            "predict", "submission"
        ]

        for s in stages:
            run_stage(s, cfg, resume=False)

        # Verification of final output artifacts
        assert (cfg.evaluation_dir / "blocking_benchmark.json").exists()
        assert (cfg.evaluation_dir / "model_validation.json").exists()
        assert (cfg.predictions_dir / "raw_predictions.parquet").exists()
        assert (cfg.predictions_dir / "entity_matches.csv").exists()
        assert (cfg.predictions_dir / "submission.csv").exists()

        df_sub = pd.read_csv(cfg.predictions_dir / "submission.csv")
        assert "source1_entity_id" in df_sub.columns
        assert "matched_entity_ids" in df_sub.columns
