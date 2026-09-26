# Evaluation & Validation Framework

Model performance is evaluated across four distinct layers in `08_validate_model.py`:

## 1. Candidate Generation Layer
- **Candidate Recall**: Proportion of true ground truth pairs retrieved during adaptive blocking.
- **Candidate Volume Metrics**: Mean, median, P95, and P99 candidate count per Source 1 entity.

## 2. Pairwise Classifier Layer
- **Precision, Recall, F1 Score** at optimal probability decision threshold.
- **ROC-AUC & PR-AUC** on pairwise predictions.

## 3. Entity-Level Ranking Layer
- **Recall@1**: Proportion of S1 entities where top-ranked prediction is a true match.
- **Recall@5**: Proportion of S1 entities where true match appears in top 5 predictions.
- **Recall@10**: Proportion of S1 entities where true match appears in top 10 predictions.

## 4. Set-Level & Subgroup Layer
- **Set Precision, Set Recall, Set F1**: Evaluating entity resolution predictions allowing zero, single, or multiple matched entities.
- **Subgroup Evaluations**: Performance broken down by cross-script entities, missing-address entities, generic-name entities, and hard negative candidates.
