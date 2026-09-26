# Implementation Status Matrix

| Component | Implemented | Synthetic Tested | Real Dataset Tested | Notes |
| --- | --- | --- | --- | --- |
| Pipeline Configuration & Memory Logger | YES | YES | NO | Configurable pathing, RSS monitoring, GC triggers |
| Stage Checkpointing & Manifest Manager | YES | YES | NO | Atomic JSON manifest tracking per chunk |
| Chunked Preprocessor (01_preprocess.py) | YES | YES | NO | Compact internal integer IDs, disk Parquet chunks |
| Disk Inverted Index Builder (02_build_indexes.py) | YES | YES | NO | Streaming DuckDB tables for exact, core, rare tokens, numeric |
| Blocking Benchmark (03_blocking_benchmark.py) | YES | YES | NO | Evaluates 10 views; outputs JSON/CSV/MD reports |
| Adaptive Candidate Generator (04_generate_candidates.py) | YES | YES | NO | EvidenceProfiler, BlockingPolicy, candidate overflow fallback |
| Training Pair Builder (05_build_training_pairs.py) | YES | YES | NO | Ground truth labeling + hard negative candidates |
| Memory-Safe Feature Engineering (06_build_features.py) | YES | YES | NO | Compact `float32`/`int8` string & transliteration features |
| Pairwise Model Training (07_train_model.py) | YES | YES | NO | Grouped entity splits, XGBoost / LightGBM models |
| Model Validation Suite (08_validate_model.py) | YES | YES | NO | Candidate recall, pair metrics, Recall@k, subgroup stats |
| Test Candidate Generator (09_generate_test_candidates.py) | YES | YES | NO | Candidate generation without ground truth access |
| Predictor & Entity Matching (10_predict.py) | YES | YES | NO | Optimal threshold, zero / one-to-many match set logic |
| Submission Adapter (11_build_submission.py) | YES | YES | NO | Formats prediction CSV; marks unverified official format |
| Colab CLI Runner (run_pipeline_colab.py) | YES | YES | NO | Stage-by-stage CLI orchestrator |
| Colab Setup Notebook (colab/01_setup.ipynb) | YES | YES | NO | Colab notebook template |

*Note: As instructed, Real Dataset Tested is marked NO for all components because execution was performed against synthetic data fixtures.*
