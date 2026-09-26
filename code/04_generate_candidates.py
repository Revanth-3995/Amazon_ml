"""
Adaptive Multi-View Candidate Generation Module.
Constructs evidence profiles per Source1 entity, dynamically selects blocking views, handles candidate limits
and overflow fallback mechanisms, records evidence tags, and streams candidate chunks to disk safely.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import duckdb
import pandas as pd

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import get_rss_mb, log_disk_usage, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("04_generate_candidates")


class EvidenceProfile:
    """Holds profile information for a Source 1 record."""

    def __init__(self, record: Dict[str, Any], rare_tokens_set: Set[str]):
        self.internal_id = record["internal_id"]
        self.entity_id = record["entity_id"]
        self.normalized_name = record["normalized_name"]
        self.core_name = record["core_name"]
        self.normalized_address = record.get("normalized_address", "")
        self.country = record.get("country", "")
        self.name_script = record.get("name_script", "Latin")
        self.translit_name = record.get("translit_name", self.normalized_name)

        self.name_tokens = record.get("name_tokens", "").split() if record.get("name_tokens") else []
        self.address_tokens = record.get("address_tokens", "").split() if record.get("address_tokens") else []
        self.numeric_tokens = record.get("numeric_tokens", "").split() if record.get("numeric_tokens") else []

        self.has_name = bool(self.normalized_name)
        self.has_address = bool(self.normalized_address)
        self.has_country = bool(self.country)
        self.has_numeric = bool(self.numeric_tokens)

        # Identify rare tokens
        self.rare_name_tokens = [t for t in self.name_tokens if t in rare_tokens_set]
        self.rare_address_tokens = [t for t in self.address_tokens if t in rare_tokens_set]

        # Generic name score: if name has no rare tokens and name_tokens length <= 2
        self.is_generic_name = (len(self.rare_name_tokens) == 0 and len(self.name_tokens) <= 2)
        self.cross_script_possible = (self.name_script != "Latin")


class BlockingPolicy:
    """Selects dynamic blocking views based on EvidenceProfile and handles overflow rules."""

    def __init__(self, max_candidates_per_entity: int = 500):
        self.max_candidates_per_entity = max_candidates_per_entity

    def select_views(self, profile: EvidenceProfile) -> List[str]:
        """Select active blocking views for this profile."""
        views = []

        if profile.has_name:
            views.append("exact_name")
            if profile.core_name:
                views.append("core_name")

        # Use rare name tokens if not overly generic
        if profile.rare_name_tokens and not profile.is_generic_name:
            views.append("rare_name_token")

        if profile.rare_address_tokens:
            views.append("rare_address_token")

        if profile.has_numeric:
            views.append("numeric_token")

        if profile.cross_script_possible and profile.translit_name:
            views.append("translit_name")

        # Fallback if no specific token views selected
        if not views and profile.has_country:
            views.append("country")

        return views


def generate_candidates(config: Config, resume: bool = True) -> None:
    """Generate candidates for all S1 entities in chunks using DuckDB disk indexes."""
    config.ensure_directories()
    manifest = ManifestManager(config.candidates_dir / "manifest.json")

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET max_memory='{config.memory_warning_mb}MB'")

    idx_dir = config.indexes_dir
    proc_dir = config.processed_dir

    # Load rare tokens into set for profiler
    freq_path = idx_dir / "token_frequencies.parquet"
    if not freq_path.exists():
        logger.error(f"Token frequency table {freq_path} not found. Run 02_build_indexes.py first.")
        return

    # High document frequency tokens are not rare
    df_freq = pd.read_parquet(freq_path)
    rare_tokens_set = set(df_freq[df_freq["doc_freq"] <= 50]["token"].tolist())

    # Create DuckDB index views
    conn.execute(f"CREATE VIEW idx_exact AS SELECT * FROM read_parquet('{idx_dir / 'exact_name_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_core AS SELECT * FROM read_parquet('{idx_dir / 'core_name_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_rare_name AS SELECT * FROM read_parquet('{idx_dir / 'rare_name_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_rare_addr AS SELECT * FROM read_parquet('{idx_dir / 'rare_address_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_numeric AS SELECT * FROM read_parquet('{idx_dir / 'numeric_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_translit AS SELECT * FROM read_parquet('{idx_dir / 'translit_name_index.parquet'}')")

    policy = BlockingPolicy(max_candidates_per_entity=config.max_candidates_per_entity)

    # Process all S1 chunks
    s1_files = sorted(list(proc_dir.glob("*source1_chunk_*.parquet")))
    s1_files = [f for f in s1_files if not f.name.startswith("id_map_")]

    stage_name = "generate_candidates"
    manifest.mark_stage_started(stage_name)

    for chunk_file in s1_files:
        chunk_id = chunk_file.stem
        out_parquet = config.candidates_dir / f"candidates_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed candidates chunk {chunk_id}")
            continue

        logger.info(f"Generating candidates for S1 chunk: {chunk_file.name}")
        df_s1 = pd.read_parquet(chunk_file)

        all_candidate_rows = []

        for _, row in df_s1.iterrows():
            profile = EvidenceProfile(row.to_dict(), rare_tokens_set)
            views = policy.select_views(profile)

            candidates_map: Dict[Tuple[int, str], List[str]] = {}

            # Execute queries for each selected view
            if "exact_name" in views and profile.normalized_name:
                exact_matches = conn.execute(
                    "SELECT internal_id, entity_id FROM idx_exact WHERE normalized_name = ?",
                    [profile.normalized_name]
                ).fetchall()
                for target_int_id, target_ent_id in exact_matches:
                    if target_int_id != profile.internal_id:
                        key = (target_int_id, target_ent_id)
                        candidates_map.setdefault(key, []).append("exact_name")

            if "core_name" in views and profile.core_name:
                core_matches = conn.execute(
                    "SELECT internal_id, entity_id FROM idx_core WHERE core_name = ?",
                    [profile.core_name]
                ).fetchall()
                for target_int_id, target_ent_id in core_matches:
                    if target_int_id != profile.internal_id:
                        key = (target_int_id, target_ent_id)
                        candidates_map.setdefault(key, []).append("core_name")

            if "rare_name_token" in views and profile.rare_name_tokens:
                for tok in profile.rare_name_tokens:
                    tok_matches = conn.execute(
                        "SELECT internal_id, entity_id FROM idx_rare_name WHERE token = ?",
                        [tok]
                    ).fetchall()
                    for target_int_id, target_ent_id in tok_matches:
                        if target_int_id != profile.internal_id:
                            key = (target_int_id, target_ent_id)
                            candidates_map.setdefault(key, []).append("rare_name_token")

            if "rare_address_token" in views and profile.rare_address_tokens:
                for tok in profile.rare_address_tokens:
                    tok_matches = conn.execute(
                        "SELECT internal_id, entity_id FROM idx_rare_addr WHERE token = ?",
                        [tok]
                    ).fetchall()
                    for target_int_id, target_ent_id in tok_matches:
                        if target_int_id != profile.internal_id:
                            key = (target_int_id, target_ent_id)
                            candidates_map.setdefault(key, []).append("rare_address_token")

            if "numeric_token" in views and profile.numeric_tokens:
                for num_tok in profile.numeric_tokens:
                    num_matches = conn.execute(
                        "SELECT internal_id, entity_id FROM idx_numeric WHERE numeric_token = ?",
                        [num_tok]
                    ).fetchall()
                    for target_int_id, target_ent_id in num_matches:
                        if target_int_id != profile.internal_id:
                            key = (target_int_id, target_ent_id)
                            candidates_map.setdefault(key, []).append("numeric_token")

            if "translit_name" in views and profile.translit_name:
                translit_matches = conn.execute(
                    "SELECT internal_id, entity_id FROM idx_translit WHERE translit_name = ?",
                    [profile.translit_name]
                ).fetchall()
                for target_int_id, target_ent_id in translit_matches:
                    if target_int_id != profile.internal_id:
                        key = (target_int_id, target_ent_id)
                        candidates_map.setdefault(key, []).append("translit_name")

            # Check candidate overflow
            overflow = 0
            if len(candidates_map) > config.max_candidates_per_entity:
                overflow = 1
                logger.debug(f"Candidate explosion for {profile.entity_id}: {len(candidates_map)} candidates. Triggering fallback.")
                # Fallback: keep top candidates that match core_name or rare address tokens
                filtered_map = {}
                for (target_int_id, target_ent_id), ev_views in candidates_map.items():
                    if "core_name" in ev_views or "rare_address_token" in ev_views or "exact_name" in ev_views:
                        filtered_map[(target_int_id, target_ent_id)] = ev_views

                # If still empty or over limit, cap at max_candidates_per_entity
                if filtered_map:
                    candidates_map = dict(list(filtered_map.items())[:config.max_candidates_per_entity])
                else:
                    candidates_map = dict(list(candidates_map.items())[:config.max_candidates_per_entity])

            # Flatten to candidate records
            for (target_int_id, target_ent_id), ev_views in candidates_map.items():
                all_candidate_rows.append({
                    "s1_internal_id": profile.internal_id,
                    "s1_entity_id": profile.entity_id,
                    "target_internal_id": target_int_id,
                    "target_entity_id": target_ent_id,
                    "blocking_views": ",".join(sorted(set(ev_views))),
                    "num_blocking_views": len(set(ev_views)),
                    "overflow": overflow,
                })

        df_cands = pd.DataFrame(all_candidate_rows)
        if not df_cands.empty:
            df_cands.to_parquet(out_parquet, index=False)
        else:
            # Empty candidate frame schema
            pd.DataFrame(columns=[
                "s1_internal_id", "s1_entity_id", "target_internal_id", "target_entity_id",
                "blocking_views", "num_blocking_views", "overflow"
            ]).to_parquet(out_parquet, index=False)

        manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), len(df_cands))
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
