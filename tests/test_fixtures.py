"""
Tests for synthetic data fixture generator.
"""

import tempfile
from pathlib import Path
import pandas as pd
from tests.fixtures.synthetic_data import generate_synthetic_dataset


def test_synthetic_data_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        dataset_files = generate_synthetic_dataset(output_dir)

        for key, path in dataset_files.items():
            assert path.exists(), f"File {path} does not exist"
            assert path.stat().st_size > 0, f"File {path} is empty"

        df_s1 = pd.read_csv(dataset_files["train_s1"], sep="\t")
        df_gt = pd.read_csv(dataset_files["train_gt"], sep="\t")

        assert len(df_s1) > 10
        assert "entity_id" in df_s1.columns
        assert "business_name" in df_s1.columns
        assert "business_address" in df_s1.columns
        assert "country" in df_s1.columns

        assert "source1_entity_id" in df_gt.columns
        assert "matched_entity_ids" in df_gt.columns
