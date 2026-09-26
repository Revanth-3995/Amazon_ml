"""
Integration test for preprocessing and disk-backed indexing on synthetic test data.
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


def test_preprocess_and_indexes_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "dataset"
        generate_synthetic_dataset(dataset_dir)

        cfg = Config(project_root=str(tmp_path), chunk_size=10)
        cfg.ensure_directories()

        # 1. Run Preprocessing
        run_preprocessing(cfg, resume=False)

        # Check processed files generated
        processed_files = list(cfg.processed_dir.glob("*.parquet"))
        assert len(processed_files) > 0, "No processed parquet files were generated"

        # Read one chunk to verify columns
        sample_chunk = [f for f in processed_files if not f.name.startswith("id_map_")][0]
        df_chunk = pd.read_parquet(sample_chunk)
        assert "internal_id" in df_chunk.columns
        assert "entity_id" in df_chunk.columns
        assert "normalized_name" in df_chunk.columns
        assert "core_name" in df_chunk.columns
        assert "country" in df_chunk.columns

        # Verify ID mapping file
        id_map_files = list(cfg.processed_dir.glob("id_map_*.parquet"))
        assert len(id_map_files) > 0, "No ID map files generated"
        df_id = pd.read_parquet(id_map_files[0])
        assert "internal_id" in df_id.columns
        assert "original_entity_id" in df_id.columns

        # 2. Run Index Building
        build_indexes(cfg, resume=False)

        index_files = list(cfg.indexes_dir.glob("*.parquet"))
        assert len(index_files) >= 5, "Index building failed to generate parquet index files"

        # Verify token frequency table
        freq_path = cfg.indexes_dir / "token_frequencies.parquet"
        assert freq_path.exists()
        df_freq = pd.read_parquet(freq_path)
        assert "token" in df_freq.columns
        assert "doc_freq" in df_freq.columns
