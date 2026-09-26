# System Architecture

## Processing Pipeline Overview

```
RAW TSV
  ↓
CHUNKED PREPROCESSING (01_preprocess.py)
  ↓
DISK-BACKED NORMALIZED PARQUET DATA
  ↓
DISK-BACKED INVERTED INDEXES (02_build_indexes.py)
  ↓
BLOCKING BENCHMARK (03_blocking_benchmark.py)
  ↓
ADAPTIVE CANDIDATE GENERATION (04_generate_candidates.py)
  ↓
TRAINING PAIRS GENERATION (05_build_training_pairs.py)
  ↓
FEATURE ENGINEERING (06_build_features.py)
  ↓
PAIRWISE MODEL TRAINING (07_train_model.py)
  ↓
MODEL VALIDATION (08_validate_model.py)
  ↓
TEST CANDIDATES & PREDICTIONS (09_generate_test_candidates.py, 10_predict.py)
  ↓
SUBMISSION FORMATTING (11_build_submission.py)
```

## Internal Data Flow & Storage
- **Compact Integer Entity Mapping**: Entity string IDs (e.g., `S1-100234`) are mapped to contiguous 32-bit internal integer IDs (`internal_id`). String IDs are restored only at final prediction output.
- **DuckDB Integration**: Analytical queries, unnested token frequency calculations, and multi-view joins are executed directly on disk-backed Parquet files via DuckDB engine.
- **Stage Manifest Checkpointing**: Every stage records atomic status in `manifest.json`. If execution disconnects or crashes, subsequent runs resume seamlessly from the last uncompleted chunk.
