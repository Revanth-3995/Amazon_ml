"""
Configuration module for Amazon ML Challenge Business Entity Resolution.
All paths and memory parameters are configurable.
"""

import os
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, project_root: str = None, chunk_size: int = 50000):
        # Base paths
        if project_root is None:
            project_root = os.environ.get(
                "PROJECT_ROOT",
                str(Path(__file__).resolve().parent.parent)
            )
        self.project_root = Path(project_root).resolve()

        # Directories
        self.code_dir = self.project_root / "code"
        self.config_dir = self.project_root / "config"
        self.dataset_dir = self.project_root / "dataset"
        self.train_dataset_dir = self.dataset_dir / "train"
        self.test_dataset_dir = self.dataset_dir / "test"

        self.processed_dir = self.project_root / "processed"
        self.indexes_dir = self.project_root / "indexes"
        self.candidates_dir = self.project_root / "candidates"
        self.features_dir = self.project_root / "features"
        self.models_dir = self.project_root / "models"
        self.predictions_dir = self.project_root / "predictions"
        self.evaluation_dir = self.project_root / "evaluation"
        self.logs_dir = self.project_root / "logs"
        self.work_dir = self.project_root / "work"

        # Processing & Memory Parameters
        self.chunk_size = chunk_size
        self.memory_target_mb = 2000
        self.memory_warning_mb = 4000
        self.memory_ceiling_mb = 8000

        # Candidate & Blocking Parameters
        self.max_candidates_per_entity = 500
        self.min_token_freq = 1
        self.max_token_freq_ratio = 0.02  # Tokens appearing in >2% docs are generic
        self.max_candidates_per_block = 1000

        # Transliteration
        self.enable_transliteration = True

        # Random seed
        self.seed = 42

    def ensure_directories(self) -> None:
        """Ensure all output directories exist."""
        for d in [
            self.code_dir,
            self.config_dir,
            self.dataset_dir,
            self.train_dataset_dir,
            self.test_dataset_dir,
            self.processed_dir,
            self.indexes_dir,
            self.candidates_dir,
            self.features_dir,
            self.models_dir,
            self.predictions_dir,
            self.evaluation_dir,
            self.logs_dir,
            self.work_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config parameters to dictionary."""
        return {
            "project_root": str(self.project_root),
            "chunk_size": self.chunk_size,
            "memory_target_mb": self.memory_target_mb,
            "memory_warning_mb": self.memory_warning_mb,
            "memory_ceiling_mb": self.memory_ceiling_mb,
            "max_candidates_per_entity": self.max_candidates_per_entity,
            "min_token_freq": self.min_token_freq,
            "max_token_freq_ratio": self.max_token_freq_ratio,
            "max_candidates_per_block": self.max_candidates_per_block,
            "enable_transliteration": self.enable_transliteration,
            "seed": self.seed,
        }
