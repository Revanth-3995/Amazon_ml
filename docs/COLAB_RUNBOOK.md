# Google Colab Execution Runbook

This guide details how to execute the Entity Resolution pipeline in Google Colab step-by-step.

## 1. Google Drive Setup
1. Mount Google Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Upload or clone the repository to:
`/content/drive/MyDrive/amazon_ml_challenge`

3. Expected directory structure on Drive:
```
/content/drive/MyDrive/amazon_ml_challenge/
├── code/
├── config/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
```

## 2. Environment Verification
Install required dependencies:
```bash
pip install pandas pyarrow duckdb xgboost lightgbm scikit-learn psutil pytest
```
Run synthetic test suite to verify installation:
```bash
cd /content/drive/MyDrive/amazon_ml_challenge
PYTHONPATH=. pytest tests/
```

## 3. Step-by-Step Stage Execution

### Step A: Preprocessing
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage preprocess --chunk-size 50000
```
- Outputs partitioned Parquet chunks to `processed/`.
- Check RAM log in output to verify RSS remains < 2 GB.

### Step B: Build Indexes
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage indexes
```
- Builds DuckDB inverted index tables in `indexes/`.

### Step C: Blocking Benchmark
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage benchmark
```
- Outputs `evaluation/blocking_benchmark_report.md`. Review candidate recall and candidates/S1 volume statistics.

### Step D: Candidate Generation
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage candidates
```
- Outputs adaptive candidate Parquet chunks to `candidates/`.

### Step E: Training Pairs & Features
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage pairs
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage features
```
- Generates compact `float32`/`int8` feature vectors in `features/`.

### Step F: Model Training & Validation
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage train
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage validate
```
- Outputs trained model to `models/` and validation metrics to `evaluation/model_validation_report.md`.

### Step G: Test Candidate Generation, Prediction & Submission
```bash
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage test_candidates
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage predict
python code/run_pipeline_colab.py --project-root /content/drive/MyDrive/amazon_ml_challenge --stage submission
```
- Final predictions written to `predictions/submission.csv`.

## Resuming After Colab Disconnections
If Colab disconnects, simply re-run the same command. The system inspects `manifest.json` in each stage directory and skips completed chunks automatically.
