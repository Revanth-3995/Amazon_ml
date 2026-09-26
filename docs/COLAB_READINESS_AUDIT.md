# Colab Readiness & Architectural Audit

This audit evaluates the Business Entity Resolution pipeline specifically for Google Colab (~12.7 GB system RAM, target RSS < 2 GB, preferred peak < 4 GB, ceiling < 8 GB) on multi-million row datasets stored on Google Drive.

## 1. Audit Checkpoints & Design Safeguards

### Memory & Scale Protections
1. **Zero Full-DataFrame Reads**: Raw files are processed exclusively via chunked iterators (`pd.read_csv(..., chunksize=50000)`).
2. **Compact Integer Entity IDs**: String entity IDs are converted to 32-bit internal integer IDs (`internal_id`) across all source datasets (`01_preprocess.py`), keeping memory and disk allocations minimal.
3. **No Unconstrained Dictionary Allocations**: `06_build_features.py` queries DuckDB on-demand per chunk for only the active `internal_id`s in that chunk, freeing RAM (`gc.collect()`) after writing each feature Parquet chunk.
4. **Candidate Explosion Safety**: Inverted indexes enforce document frequency and block size caps (`max_candidates_per_block`). Per-entity candidate generation enforces `max_candidates_per_entity` with evidence-based rank filtering.
5. **Streaming Benchmark Queries**: Benchmark strategies (`03_blocking_benchmark.py`) aggregate recall and volume metrics via DuckDB streaming SQL without materializing full candidate joins in RAM.
6. **Optimized Numeric Types**: Feature tables use `float32`, `int8`, `int16`, and `int32` dtypes to prevent accidental `float64` memory inflation.
7. **No Hardcoded Row Counts or Countries**: Dataset dimensions and country codes are dynamically discovered during chunked execution.

## 2. Component Readiness Matrix

| Component | Memory risk | Disk risk | Checkpoint support | Real-data status | Fix applied |
| --- | --- | --- | --- | --- | --- |
| Config & Path Manager (`config/default_config.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Exposed `max_candidates_per_block` and memory limits |
| Memory & RSS Monitor (`utils/memory.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Added psutil / /proc fallback and automatic `gc.collect()` |
| Stage Checkpoint Manager (`utils/manifest.py`) | LOW | LOW | YES | NOT YET VALIDATED ON REAL DATA | Atomic JSON manifest updates per chunk |
| Chunked Preprocessor (`code/01_preprocess.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | Globally unique internal integer IDs & chunked Parquet outputs |
| Disk Inverted Index Builder (`code/02_build_indexes.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | Separated train/test target scopes & capped rare token posting lists |
| Blocking Benchmark (`code/03_blocking_benchmark.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Dynamic target source filtering and streaming DuckDB aggregations |
| Adaptive Candidate Generator (`code/04_generate_candidates.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | Target source filtering, EvidenceProfile, and candidate overflow fallback |
| Training Pair Generator (`code/05_build_training_pairs.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | Ground truth pair preservation & hard-negative candidate labeling |
| Memory-Safe Feature Builder (`code/06_build_features.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | On-demand chunk ID lookups, `float32`/`int8` dtypes, and explicit GC |
| Pairwise Model Trainer (`code/07_train_model.py`) | MEDIUM | LOW | N/A | NOT YET VALIDATED ON REAL DATA | GroupShuffleSplit entity splitting (preventing leakage) |
| Model Validation Suite (`code/08_validate_model.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Evaluated Recall@k against total dataset ground truth S1 count |
| Test Candidate Generator (`code/09_generate_test_candidates.py`) | LOW | MEDIUM | YES | NOT YET VALIDATED ON REAL DATA | Isolated target sources (`test_source2/3`) & zero ground truth access |
| Predictor & Entity Matcher (`code/10_predict.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Optimal validation probability thresholding & 0/1/many set logic |
| Submission Adapter (`code/11_build_submission.py`) | LOW | LOW | N/A | NOT YET VALIDATED ON REAL DATA | Formatted CSV predictions; marked unverified official format |
| Pipeline Orchestrator (`code/run_pipeline_colab.py`) | LOW | LOW | YES | NOT YET VALIDATED ON REAL DATA | Multi-stage CLI runner with memory logging before/after stages |

*Disclaimer: Real-data status is marked 'NOT YET VALIDATED ON REAL DATA' for all components because execution was performed against synthetic data fixtures.*
