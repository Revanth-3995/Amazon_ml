"""
Memory-Safe Feature Engineering Module.
Calculates compact numeric features (string similarity, token jaccard/overlap, numeric overlap, script,
transliteration, blocking evidence) chunk-by-chunk using optimized float32/int32/int8 dtypes
and chunk-specific DuckDB record lookups to guarantee minimal RAM usage (< 2 GB RSS).
"""

import argparse
import difflib
import gc
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

import duckdb
import numpy as np
import pandas as pd

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import get_rss_mb, log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("06_build_features")


def compute_string_similarity(str1: str, str2: str) -> float:
    """Compute string similarity using SequenceMatcher ratio."""
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    return float(difflib.SequenceMatcher(None, str1, str2).ratio())


def compute_token_jaccard(tokens1_str: str, tokens2_str: str) -> float:
    """Compute Jaccard similarity between token strings."""
    if not tokens1_str or not tokens2_str:
        return 0.0
    s1 = set(tokens1_str.split())
    s2 = set(tokens2_str.split())
    if not s1 or not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection / union) if union > 0 else 0.0


def compute_token_overlap(tokens1_str: str, tokens2_str: str) -> float:
    """Compute token overlap ratio (intersection / min(len1, len2))."""
    if not tokens1_str or not tokens2_str:
        return 0.0
    s1 = set(tokens1_str.split())
    s2 = set(tokens2_str.split())
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    return float(len(s1.intersection(s2)) / min_len) if min_len > 0 else 0.0


def build_features_for_chunk(
    df_pairs: pd.DataFrame,
    processed_map: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """Compute compact feature matrix for a chunk of candidate pairs."""
    feature_rows = []

    for _, row in df_pairs.iterrows():
        s1_id = row["s1_internal_id"]
        target_id = row["target_internal_id"]

        r1 = processed_map.get(s1_id, {})
        r2 = processed_map.get(target_id, {})

        norm_name1 = r1.get("normalized_name", "")
        norm_name2 = r2.get("normalized_name", "")

        core_name1 = r1.get("core_name", "")
        core_name2 = r2.get("core_name", "")

        norm_addr1 = r1.get("normalized_address", "")
        norm_addr2 = r2.get("normalized_address", "")

        name_tokens1 = r1.get("name_tokens", "")
        name_tokens2 = r2.get("name_tokens", "")

        addr_tokens1 = r1.get("address_tokens", "")
        addr_tokens2 = r2.get("address_tokens", "")

        num_tokens1 = r1.get("numeric_tokens", "")
        num_tokens2 = r2.get("numeric_tokens", "")

        c1 = r1.get("country", "")
        c2 = r2.get("country", "")

        script1 = r1.get("name_script", "Latin")
        script2 = r2.get("name_script", "Latin")

        translit1 = r1.get("translit_name", norm_name1)
        translit2 = r2.get("translit_name", norm_name2)

        blocking_views_str = str(row.get("blocking_views", ""))
        b_views = set(blocking_views_str.split(",")) if blocking_views_str else set()

        feat = {
            "s1_internal_id": np.int32(s1_id),
            "target_internal_id": np.int32(target_id),
            "s1_entity_id": str(row["s1_entity_id"]),
            "target_entity_id": str(row["target_entity_id"]),
            "label": np.int8(row.get("label", 0)),

            # Name features
            "name_exact": np.int8(1 if norm_name1 and norm_name1 == norm_name2 else 0),
            "core_name_exact": np.int8(1 if core_name1 and core_name1 == core_name2 else 0),
            "name_similarity": np.float32(compute_string_similarity(norm_name1, norm_name2)),
            "core_name_similarity": np.float32(compute_string_similarity(core_name1, core_name2)),
            "name_token_jaccard": np.float32(compute_token_jaccard(name_tokens1, name_tokens2)),
            "name_token_overlap": np.float32(compute_token_overlap(name_tokens1, name_tokens2)),

            # Address features
            "address_exact": np.int8(1 if norm_addr1 and norm_addr1 == norm_addr2 else 0),
            "address_similarity": np.float32(compute_string_similarity(norm_addr1, norm_addr2)),
            "address_token_jaccard": np.float32(compute_token_jaccard(addr_tokens1, addr_tokens2)),
            "address_token_overlap": np.float32(compute_token_overlap(addr_tokens1, addr_tokens2)),
            "numeric_overlap": np.float32(compute_token_overlap(num_tokens1, num_tokens2)),

            # Country feature
            "country_match": np.int8(1 if c1 and c1 == c2 else (0 if c1 and c2 else -1)),

            # Script features
            "same_script": np.int8(1 if script1 == script2 else 0),
            "cross_script": np.int8(1 if script1 != script2 and (script1 != "Latin" or script2 != "Latin") else 0),

            # Transliteration features
            "translit_name_similarity": np.float32(compute_string_similarity(translit1, translit2)),
            "translit_name_exact": np.int8(1 if translit1 and translit1 == translit2 else 0),

            # Blocking evidence features
            "num_blocking_views": np.int16(len(b_views)),
            "has_name_block": np.int8(1 if ("exact_name" in b_views or "core_name" in b_views or "rare_name_token" in b_views) else 0),
            "has_address_block": np.int8(1 if "rare_address_token" in b_views else 0),
            "has_numeric_block": np.int8(1 if "numeric_token" in b_views else 0),
            "has_translit_block": np.int8(1 if "translit_name" in b_views else 0),

            # Quality features
            "address_missing": np.int8(1 if not norm_addr1 or not norm_addr2 else 0),
            "overflow": np.int8(row.get("overflow", 0)),
        }
        feature_rows.append(feat)

    return pd.DataFrame(feature_rows)


def build_features(config: Config, is_test: bool = False, resume: bool = True) -> None:
    """Build features for training pairs or test candidate chunks."""
    config.ensure_directories()

    mode = "test" if is_test else "train"
    manifest = ManifestManager(config.features_dir / f"manifest_{mode}.json")

    stage_name = f"build_features_{mode}"
    manifest.mark_stage_started(stage_name)

    proc_files = [
        f for f in config.processed_dir.glob("*.parquet")
        if not f.name.startswith("id_map_") and not f.name.startswith("training_pairs_")
    ]
    if not proc_files:
        logger.error("No processed source files found.")
        return

    file_list_str = ", ".join([f"'{str(f)}'" for f in proc_files])

    # Find candidate/pair files
    if is_test:
        input_files = sorted(list(config.candidates_dir.glob("candidates_test_*.parquet")))
    else:
        input_files = sorted(list(config.processed_dir.glob("training_pairs_*.parquet")))

    if not input_files:
        logger.warning(f"No input pair files found for mode {mode}.")
        return

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"CREATE VIEW processed_data AS SELECT * FROM read_parquet([{file_list_str}])")

    for in_path in input_files:
        chunk_id = in_path.stem
        out_parquet = config.features_dir / f"features_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed feature chunk {chunk_id}")
            continue

        df_pairs = pd.read_parquet(in_path)
        if df_pairs.empty:
            df_empty = pd.DataFrame()
            df_empty.to_parquet(out_parquet, index=False)
            manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), 0)
            continue

        # Extract only the required internal_ids for this chunk to keep RAM minimal
        req_ids = set(df_pairs["s1_internal_id"].tolist()).union(set(df_pairs["target_internal_id"].tolist()))
        req_ids_list = list(req_ids)

        # Query DuckDB for only the required record attributes
        conn.register("req_ids_tbl", pd.DataFrame({"internal_id": req_ids_list}))
        df_chunk_records = conn.execute("""
            SELECT p.* FROM processed_data p
            JOIN req_ids_tbl r ON p.internal_id = r.internal_id
        """).fetchdf()
        conn.unregister("req_ids_tbl")

        processed_map = {row["internal_id"]: row.to_dict() for _, row in df_chunk_records.iterrows()}

        df_feats = build_features_for_chunk(df_pairs, processed_map)
        df_feats.to_parquet(out_parquet, index=False)

        # Clear chunk memory
        del df_chunk_records
        del processed_map
        gc.collect()

        manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), len(df_feats))
        log_memory(f"Built features chunk {chunk_id}")

    conn.close()
    manifest.mark_stage_completed(stage_name)
    log_disk_usage(config.features_dir, f"Features Directory ({mode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Compact Pair Features")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--is-test", action="store_true", help="Build features for test candidates")
    parser.add_argument("--no-resume", action="store_true", help="Force rebuild features")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    build_features(cfg, is_test=args.is_test, resume=not args.no_resume)
