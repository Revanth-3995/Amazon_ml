"""
Integration test for blocking benchmark and adaptive candidate generation on synthetic test data.
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

benchmark_mod = importlib.import_module("code.03_blocking_benchmark")
run_blocking_benchmark = benchmark_mod.run_blocking_benchmark

candidates_mod = importlib.import_module("code.04_generate_candidates")
generate_candidates = candidates_mod.generate_candidates


def test_blocking_benchmark_and_candidate_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "dataset"
        generate_synthetic_dataset(dataset_dir)

        cfg = Config(project_root=str(tmp_path), chunk_size=20)
        cfg.ensure_directories()

        # Step 1 & 2 setup
        run_preprocessing(cfg, resume=False)
        build_indexes(cfg, resume=False)

        # Step 3: Run Blocking Benchmark
        bench_res = run_blocking_benchmark(cfg)
        assert "results" in bench_res
        assert len(bench_res["results"]) == 10

        # Verify output report files
        assert (cfg.evaluation_dir / "blocking_benchmark.json").exists()
        assert (cfg.evaluation_dir / "blocking_benchmark.csv").exists()
        assert (cfg.evaluation_dir / "blocking_benchmark_report.md").exists()

        # Step 4: Run Candidate Generation
        generate_candidates(cfg, resume=False)

        cand_files = list(cfg.candidates_dir.glob("*.parquet"))
        assert len(cand_files) > 0, "Candidate parquet files were not generated"

        # Read candidate file and verify fields
        df_cands = pd.read_parquet(cand_files[0])
        assert "s1_internal_id" in df_cands.columns
        assert "s1_entity_id" in df_cands.columns
        assert "target_internal_id" in df_cands.columns
        assert "target_entity_id" in df_cands.columns
        assert "blocking_views" in df_cands.columns
        assert "num_blocking_views" in df_cands.columns
        assert "overflow" in df_cands.columns
