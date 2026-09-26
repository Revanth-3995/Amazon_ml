# Amazon ML Challenge 2026: Business Entity Resolution Project

A high-performance, memory-safe, disk-backed, and resumable Business Entity Resolution system designed specifically for Google Colab (~12.7 GB RAM constraints) and multi-million row datasets.

## Main Innovations
- **Adaptive Multi-View Blocking for Multilingual Business Entity Resolution**: Dynamically selects blocking views per entity based on Evidence Profile analysis (token uniqueness, script type, numeric tokens, missing address).
- **Evidence-Aware Pair Scoring**: Models evaluate candidate pairs using rich features, including number of independent blocking views and cross-script transliteration signals.
- **Disk-First Architecture**: Powered by DuckDB streaming SQL and Parquet chunking, maintaining RSS < 2 GB.
- **Zero & One-to-Many Set Decisions**: Ranks candidates and applies tuned probability decision thresholds.

## Structure
```
.
├── code/
│   ├── 01_preprocess.py
│   ├── 02_build_indexes.py
│   ├── 03_blocking_benchmark.py
│   ├── 04_generate_candidates.py
│   ├── 05_build_training_pairs.py
│   ├── 06_build_features.py
│   ├── 07_train_model.py
│   ├── 08_validate_model.py
│   ├── 09_generate_test_candidates.py
│   ├── 10_predict.py
│   ├── 11_build_submission.py
│   └── run_pipeline_colab.py
├── config/
│   └── default_config.py
├── colab/
│   └── 01_setup.ipynb
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MEMORY_DESIGN.md
│   ├── BLOCKING_DESIGN.md
│   ├── FEATURES.md
│   ├── EVALUATION.md
│   ├── NOVELTY.md
│   ├── COLAB_RUNBOOK.md
│   └── IMPLEMENTATION_STATUS.md
├── tests/
└── utils/
    ├── memory.py
    ├── manifest.py
    ├── text_processing.py
    └── transliteration.py
```

## Quick Start (Synthetic Testing)
```bash
pip install pandas pyarrow duckdb xgboost lightgbm scikit-learn psutil pytest
PYTHONPATH=. pytest tests/
```

## Running on Google Colab
See `docs/COLAB_RUNBOOK.md` and `colab/01_setup.ipynb` for step-by-step instructions.
