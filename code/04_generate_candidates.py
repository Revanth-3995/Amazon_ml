"""
Adaptive Multi-View Candidate Generation Module.
Uses set-based vectorized DuckDB SQL joins over S1 chunk Parquet files against disk-backed indexes
for fast execution on Google Drive without row-by-row point query bottlenecks.
"""

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import get_rss_mb, log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("04_generate_candidates")


def generate_candidates(config: Config, resume: bool = True) -> None:
    """Generate candidates for train S1 entities using set-based vectorized DuckDB joins."""
    config.ensure_directories()
    manifest = ManifestManager(config.candidates_dir / "manifest.json")

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET max_memory='{config.memory_warning_mb}MB'")

    idx_dir = config.indexes_dir
    proc_dir = config.processed_dir

    freq_path = idx_dir / "token_frequencies.parquet"
    if not freq_path.exists():
        logger.error(f"Token frequency table {freq_path} not found. Run 02_build_indexes.py first.")
        return

    # Create DuckDB index views filtered strictly for train target sources
    target_filter = "WHERE source_name IN ('train_source2', 'train_source3')"
    conn.execute(f"CREATE VIEW idx_exact AS SELECT * FROM read_parquet('{idx_dir / 'exact_name_index.parquet'}') {target_filter}")
    conn.execute(f"CREATE VIEW idx_core AS SELECT * FROM read_parquet('{idx_dir / 'core_name_index.parquet'}') {target_filter}")
    conn.execute(f"CREATE VIEW idx_rare_name AS SELECT * FROM read_parquet('{idx_dir / 'rare_name_token_index.parquet'}') {target_filter}")
    conn.execute(f"CREATE VIEW idx_rare_addr AS SELECT * FROM read_parquet('{idx_dir / 'rare_address_token_index.parquet'}') {target_filter}")
    conn.execute(f"CREATE VIEW idx_numeric AS SELECT * FROM read_parquet('{idx_dir / 'numeric_token_index.parquet'}') {target_filter}")
    conn.execute(f"CREATE VIEW idx_translit AS SELECT * FROM read_parquet('{idx_dir / 'translit_name_index.parquet'}') {target_filter}")

    s1_files = sorted(list(proc_dir.glob("train_source1_chunk_*.parquet")))
    s1_files = [f for f in s1_files if not f.name.startswith("id_map_")]

    stage_name = "generate_candidates"
    manifest.mark_stage_started(stage_name)

    for chunk_file in s1_files:
        chunk_id = chunk_file.stem
        out_parquet = config.candidates_dir / f"candidates_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed candidates chunk {chunk_id}")
            continue

        logger.info(f"Vectorized candidate generation for S1 chunk: {chunk_file.name}")

        conn.execute(f"CREATE VIEW s1_chunk AS SELECT * FROM read_parquet('{str(chunk_file)}')")

        # Execute vectorized multi-view blocking join query directly in DuckDB
        conn.execute(f"""
            CREATE TEMP TABLE raw_cands AS
            -- Exact Name View
            SELECT s1.internal_id AS s1_internal_id, s1.entity_id AS s1_entity_id,
                   t.internal_id AS target_internal_id, t.entity_id AS target_entity_id,
                   'exact_name' AS blocking_view
            FROM s1_chunk s1 JOIN idx_exact t ON s1.normalized_name = t.normalized_name AND s1.internal_id != t.internal_id
            WHERE s1.normalized_name != ''

            UNION ALL

            -- Core Name View
            SELECT s1.internal_id, s1.entity_id, t.internal_id, t.entity_id, 'core_name'
            FROM s1_chunk s1 JOIN idx_core t ON s1.core_name = t.core_name AND s1.internal_id != t.internal_id
            WHERE s1.core_name != ''

            UNION ALL

            -- Rare Name Token View
            SELECT s1.internal_id, s1.entity_id, t.internal_id, t.entity_id, 'rare_name_token'
            FROM (
                SELECT internal_id, entity_id, unnest(string_split(name_tokens, ' ')) AS token
                FROM s1_chunk WHERE name_tokens != ''
            ) s1 JOIN idx_rare_name t ON s1.token = t.token AND s1.internal_id != t.internal_id

            UNION ALL

            -- Rare Address Token View
            SELECT s1.internal_id, s1.entity_id, t.internal_id, t.entity_id, 'rare_address_token'
            FROM (
                SELECT internal_id, entity_id, unnest(string_split(address_tokens, ' ')) AS token
                FROM s1_chunk WHERE address_tokens != ''
            ) s1 JOIN idx_rare_addr t ON s1.token = t.token AND s1.internal_id != t.internal_id

            UNION ALL

            -- Numeric Token View
            SELECT s1.internal_id, s1.entity_id, t.internal_id, t.entity_id, 'numeric_token'
            FROM (
                SELECT internal_id, entity_id, unnest(string_split(numeric_tokens, ' ')) AS numeric_token
                FROM s1_chunk WHERE numeric_tokens != ''
            ) s1 JOIN idx_numeric t ON s1.numeric_token = t.numeric_token AND s1.internal_id != t.internal_id

            UNION ALL

            -- Transliterated Name View
            SELECT s1.internal_id, s1.entity_id, t.internal_id, t.entity_id, 'translit_name'
            FROM s1_chunk s1 JOIN idx_translit t ON s1.translit_name = t.translit_name AND s1.internal_id != t.internal_id
            WHERE s1.translit_name != ''
        """)

        # Aggregate evidence views per pair and rank candidates to handle overflow
        conn.execute(f"""
            COPY (
                WITH aggregated_cands AS (
                    SELECT s1_internal_id, s1_entity_id, target_internal_id, target_entity_id,
                           string_agg(DISTINCT blocking_view, ',') AS blocking_views,
                           COUNT(DISTINCT blocking_view) AS num_blocking_views
                    FROM raw_cands
                    GROUP BY s1_internal_id, s1_entity_id, target_internal_id, target_entity_id
                ),
                ranked_cands AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY s1_internal_id
                               ORDER BY num_blocking_views DESC, target_internal_id ASC
                           ) AS rank_idx,
                           COUNT(*) OVER (PARTITION BY s1_internal_id) AS total_entity_cands
                    FROM aggregated_cands
                )
                SELECT s1_internal_id, s1_entity_id, target_internal_id, target_entity_id,
                       blocking_views, num_blocking_views,
                       CASE WHEN total_entity_cands > {config.max_candidates_per_entity} THEN 1 ELSE 0 END AS overflow
                FROM ranked_cands
                WHERE rank_idx <= {config.max_candidates_per_entity}
            ) TO '{out_parquet}' (FORMAT PARQUET)
        """)

        conn.execute("DROP TABLE raw_cands")
        conn.execute("DROP VIEW s1_chunk")

        cand_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_parquet}')").fetchone()[0]
        manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), cand_count)
        log_memory(f"Generated candidates chunk {chunk_id}")

    conn.close()
    manifest.mark_stage_completed(stage_name)
    log_disk_usage(config.candidates_dir, "Candidates Directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Candidate Generator")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--no-resume", action="store_true", help="Force regenerate candidates")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    generate_candidates(cfg, resume=not args.no_resume)
