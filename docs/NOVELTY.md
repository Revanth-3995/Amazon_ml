# Main Project Innovations

This project introduces two core design contributions for business entity resolution under low-RAM constraints:

## 1. Adaptive Multi-View Blocking for Multilingual Business Entity Resolution
Standard blocking techniques apply fixed blocking rules (such as shared name tokens or exact country) across all records. In business entity resolution, this leads to candidate explosion for common names (e.g. "Primary Care") or zero recall for cross-script transliterated entities.

Our system builds an `EvidenceProfile` per Source 1 entity (analyzing token document frequency, script family, presence of house/unit numbers, and missing address indicators) and dynamically selects tailored blocking views. If an entity triggers a candidate explosion, the policy activates candidate overflow fallback mechanisms to constrain the candidate space using tight core-name and address token intersections.

## 2. Evidence-Aware Pair Scoring
Candidate pairs store metadata reflecting which blocking views generated the candidate pair and how many independent blocking channels confirmed the pair (`num_blocking_views`, `has_name_block`, `has_address_block`, `has_numeric_block`, `has_translit_block`). The downstream XGBoost / LightGBM classifier learns to weight evidence redundancy, enabling high precision even when individual text similarity signals are noisy or cross-script.
