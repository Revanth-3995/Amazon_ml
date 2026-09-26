"""
Integration test for training pair generation and memory-safe feature extraction.
"""

import importlib
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from config.default_config import Config
from tests.fixtures.synthetic_data import generate_synthetic_dataset

preprocess_mod = importlib.import_module("code.01_preprocess")
run_preprocessing = preprocess_mod.run_preprocessing

indexes_mod = importlib.import_module("code.02_build_indexes")
build_indexes = indexes_mod.build_indexes

candidates_mod = importlib.import_module("code.04_generate_candidates")
generate_candidates = candidates_mod.generate_candidates

pairs_mod = importlib.import_module("code.05_build_training_pairs")
build_training_pairs = pairs_mod.build_training_pairs

features_mod = importlib.import_module("code.06_build_features")
build_features = features_mod.build_features


def test_pairs_and_features_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "dataset"
        generate_synthetic_dataset(dataset_dir)

        cfg = Config(project_root=str(tmp_path), chunk_size=20)
        cfg.ensure_directories()

        # Step 1, 2 & 3 setup
        run_preprocessing(cfg, resume=False)
        build_indexes(cfg, resume=False)
        generate_candidates(cfg, resume=False)

        # Step 4A: Build Training Pairs
        build_training_pairs(cfg, resume=False)

        pair_files = list(cfg.processed_dir.glob("training_pairs_*.parquet"))
        assert len(pair_files) > 0, "No training pair parquet files generated"

        df_pairs = pd.read_parquet(pair_files[0])
        assert "s1_entity_id" in df_pairs.columns
        assert "target_entity_id" in df_pairs.columns
        assert "label" in df_pairs.columns

        # Verify ground truth labels present (both 1 and 0 hard negatives)
        labels = df_pairs["label"].unique()
        assert 1 in labels or 0 in labels

        # Step 4B: Build Features
        build_features(cfg, is_test=False, resume=False)

        feat_files = list(cfg.features_dir.glob("features_*.parquet"))
        assert len(feat_files) > 0, "No feature parquet files generated"

        df_feats = pd.read_parquet(feat_files[0])
        expected_cols = [
            "name_exact", "core_name_exact", "name_similarity", "core_name_similarity",
            "name_token_jaccard", "name_token_overlap", "address_exact", "address_similarity",
            "address_token_jaccard", "address_token_overlap", "numeric_overlap", "country_match",
            "same_script", "cross_script", "translit_name_similarity", "translit_name_exact",
            "num_blocking_views", "has_name_block", "has_address_block", "has_numeric_block",
            "has_translit_block", "address_missing", "overflow", "label"
        ]
        for col in expected_cols:
            assert col in df_feats.columns, f"Missing feature column: {col}"

        # Verify optimized compact dtypes
        assert df_feats["name_similarity"].dtype == "float32"
        assert df_feats["name_exact"].dtype == "int8"
