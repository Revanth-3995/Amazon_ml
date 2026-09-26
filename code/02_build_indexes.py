"""
Disk-backed Inverted Index Builder using DuckDB.
Builds token frequency statistics and inverted indexes (exact name, core name, rare tokens, numeric tokens, country)
separately for train and test target sources without loading complete dataset DataFrames into RAM.
"""

import argparse
import logging
from pathlib import Path

import duckdb

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("02_build_indexes")


def build_indexes(config: Config, resume: bool = True) -> None:
    """Build disk-backed inverted indexes using DuckDB streaming SQL execution."""
    config.ensure_directories()
    manifest = ManifestManager(config.indexes_dir / "manifest.json")

    stage_name = "build_indexes"
    if resume and manifest.is_stage_completed(stage_name):
        logger.info("Indexes stage already completed. Skipping.")
        return

    manifest.mark_stage_started(stage_name)

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET max_memory='{config.memory_warning_mb}MB'")
    conn.execute("SET threads=2")

    logger.info("Building token document frequencies...")

    parquet_files = [
        f for f in config.processed_dir.glob("*.parquet")
        if not f.name.startswith("id_map_") and not f.name.startswith("training_pairs_")
    ]

    if not parquet_files:
        logger.warning(f"No processed parquet files found in {config.processed_dir}. Run 01_preprocess.py first.")
        return

    file_list_str = ", ".join([f"'{str(f)}'" for f in parquet_files])
    conn.execute(f"""
        CREATE VIEW processed_data AS
        SELECT * FROM read_parquet([{file_list_str}])
    """)

    # 1. Compute Token Frequencies
    token_freq_path = config.indexes_dir / "token_frequencies.parquet"
    conn.execute(f"""
        COPY (
            WITH unnested_name_tokens AS (
                SELECT unnest(string_split(name_tokens, ' ')) AS token
                FROM processed_data
                WHERE name_tokens != ''
            ),
            unnested_addr_tokens AS (
                SELECT unnest(string_split(address_tokens, ' ')) AS token
                FROM processed_data
                WHERE address_tokens != ''
            ),
            combined_tokens AS (
                SELECT token, 'name' AS token_type FROM unnested_name_tokens
                UNION ALL
                SELECT token, 'address' AS token_type FROM unnested_addr_tokens
            )
            SELECT token, token_type, COUNT(*) AS doc_freq
            FROM combined_tokens
            WHERE length(token) >= 2
            GROUP BY token, token_type
        ) TO '{token_freq_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built token frequency table at {token_freq_path}")

    total_docs = conn.execute("SELECT COUNT(*) FROM processed_data").fetchone()[0]
    rare_threshold = max(2, int(total_docs * config.max_token_freq_ratio))
    logger.info(f"Total documents: {total_docs}. Rare token max threshold: {rare_threshold}")

    conn.execute(f"""
        CREATE VIEW rare_tokens AS
        SELECT token, token_type, doc_freq
        FROM read_parquet('{token_freq_path}')
        WHERE doc_freq <= {rare_threshold}
    """)

    # 2. Exact Normalized Name Index
    exact_name_idx_path = config.indexes_dir / "exact_name_index.parquet"
    conn.execute(f"""
        COPY (
            SELECT normalized_name, internal_id, entity_id, country, source_name
            FROM processed_data
            WHERE normalized_name != ''
        ) TO '{exact_name_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built exact name index at {exact_name_idx_path}")

    # 3. Core Name Index
    core_name_idx_path = config.indexes_dir / "core_name_index.parquet"
    conn.execute(f"""
        COPY (
            SELECT core_name, internal_id, entity_id, country, source_name
            FROM processed_data
            WHERE core_name != ''
        ) TO '{core_name_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built core name index at {core_name_idx_path}")

    # 4. Rare Name Token Index
    rare_name_idx_path = config.indexes_dir / "rare_name_token_index.parquet"
    conn.execute(f"""
        COPY (
            WITH name_tokens_split AS (
                SELECT unnest(string_split(name_tokens, ' ')) AS token, internal_id, entity_id, country, source_name
                FROM processed_data
                WHERE name_tokens != ''
            )
            SELECT t.token, t.internal_id, t.entity_id, t.country, t.source_name
            FROM name_tokens_split t
            JOIN rare_tokens r ON t.token = r.token AND r.token_type = 'name'
        ) TO '{rare_name_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built rare name token index at {rare_name_idx_path}")

    # 5. Rare Address Token Index
    rare_addr_idx_path = config.indexes_dir / "rare_address_token_index.parquet"
    conn.execute(f"""
        COPY (
            WITH addr_tokens_split AS (
                SELECT unnest(string_split(address_tokens, ' ')) AS token, internal_id, entity_id, country, source_name
                FROM processed_data
                WHERE address_tokens != ''
            )
            SELECT t.token, t.internal_id, t.entity_id, t.country, t.source_name
            FROM addr_tokens_split t
            JOIN rare_tokens r ON t.token = r.token AND r.token_type = 'address'
        ) TO '{rare_addr_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built rare address token index at {rare_addr_idx_path}")

    # 6. Numeric Token Index
    numeric_idx_path = config.indexes_dir / "numeric_token_index.parquet"
    conn.execute(f"""
        COPY (
            WITH num_tokens_split AS (
                SELECT unnest(string_split(numeric_tokens, ' ')) AS numeric_token, internal_id, entity_id, country, source_name
                FROM processed_data
                WHERE numeric_tokens != ''
            )
            SELECT numeric_token, internal_id, entity_id, country, source_name
            FROM num_tokens_split
            WHERE length(numeric_token) >= 2
        ) TO '{numeric_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built numeric token index at {numeric_idx_path}")

    # 7. Transliterated Name Index
    translit_idx_path = config.indexes_dir / "translit_name_index.parquet"
    conn.execute(f"""
        COPY (
            SELECT translit_name, internal_id, entity_id, country, source_name
            FROM processed_data
            WHERE translit_name != ''
        ) TO '{translit_idx_path}' (FORMAT PARQUET)
    """)
    logger.info(f"Built transliterated name index at {translit_idx_path}")

    conn.close()

    manifest.mark_stage_completed(stage_name, {"total_docs": total_docs, "rare_threshold": rare_threshold})
    log_memory("Build Indexes Stage Complete")
    log_disk_usage(config.indexes_dir, "Indexes Directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Disk-Backed Inverted Indexes")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--no-resume", action="store_true", help="Force rebuild indexes")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    build_indexes(cfg, resume=not args.no_resume)
