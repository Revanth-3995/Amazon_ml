# Adaptive Multi-View Blocking Design

## Dynamic Evidence Profiling
For every Source 1 entity, an `EvidenceProfile` is generated:
- `normalized_name`, `core_name`, `normalized_address`, `country`
- `name_script`, `address_script`, `cross_script_possible`
- `rare_name_tokens`, `rare_address_tokens`, `numeric_tokens`
- `is_generic_name`: True if name consists only of common/frequent tokens.

## Blocking Strategy Selection
`BlockingPolicy` dynamically selects active blocking views:
1. `exact_name`: Normalized exact string matching.
2. `core_name`: Normalized string matching excluding legal entity terms (pvt, ltd, inc, corp, llc, etc.).
3. `rare_name_token`: Inverted index matching on tokens appearing in <= 2% of documents.
4. `rare_address_token`: Inverted index matching on address tokens.
5. `numeric_token`: Inverted index matching on house/unit/door numbers.
6. `translit_name`: Cross-script transliteration index matching (e.g. Indic script -> ASCII Latin).

## Candidate Overflow Protection & Fallback
- `max_candidates_per_entity`: Enforces configurable candidate cap (default: 500).
- If candidate generation for an entity exceeds this limit:
  1. Records `overflow = 1` flag.
  2. Disallows generic unconstrained views.
  3. Filters candidate pool using tighter core name or rare address token constraints before truncation.
