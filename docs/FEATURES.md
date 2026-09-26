# Feature Engineering Specification

The feature matrix converts candidate pairs `(s1, target)` into numeric feature vectors using `06_build_features.py`:

## 1. Name Features
- `name_exact` (`int8`): Exact normalized name match.
- `core_name_exact` (`int8`): Core name match excluding legal entity suffixes.
- `name_similarity` (`float32`): SequenceMatcher similarity ratio on normalized names.
- `core_name_similarity` (`float32`): Similarity ratio on core names.
- `name_token_jaccard` (`float32`): Jaccard token set similarity.
- `name_token_overlap` (`float32`): Token overlap ratio relative to minimum token count.

## 2. Address Features
- `address_exact` (`int8`): Exact normalized address match.
- `address_similarity` (`float32`): Address string similarity.
- `address_token_jaccard` (`float32`): Address token Jaccard similarity.
- `address_token_overlap` (`float32`): Address token overlap ratio.
- `numeric_overlap` (`float32`): Overlap ratio of extracted numeric tokens (house/door/zip numbers).

## 3. Country & Script Features
- `country_match` (`int8`): 1 if countries match, 0 if mismatch, -1 if country missing.
- `same_script` (`int8`): 1 if both records use the same script family.
- `cross_script` (`int8`): 1 if records belong to different script families.

## 4. Transliteration Features
- `translit_name_similarity` (`float32`): Similarity on transliterated ASCII representations.
- `translit_name_exact` (`int8`): Exact match on transliterated names.

## 5. Blocking & Quality Indicators
- `num_blocking_views` (`int16`): Count of independent blocking views that generated this candidate pair.
- `has_name_block`, `has_address_block`, `has_numeric_block`, `has_translit_block` (`int8`): Indicators for view sources.
- `address_missing` (`int8`): 1 if address field was missing in either record.
- `overflow` (`int8`): 1 if candidate count triggered safety overflow logic.
