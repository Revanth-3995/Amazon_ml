"""
Test Candidate Generation Module.
Generates candidate pairs for test_source1 against test_source2 / test_source3 indexes without accessing ground truth.
"""

import argparse
import importlib
import logging
from pathlib import Path

import duckdb
import pandas as pd

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import log_disk_usage, log_memory

gen_cands_mod = importlib.import_module("code.04_generate_candidates")
BlockingPolicy = gen_cands_mod.BlockingPolicy
EvidenceProfile = gen_cands_mod.EvidenceProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("09_generate_test_candidates")


def generate_test_candidates(config: Config, resume: bool = True) -> None:
    """Generate candidates for test set without ground truth."""
    config.ensure_directories()
    manifest = ManifestManager(config.candidates_dir / "manifest_test.json")

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET max_memory='{config.memory_warning_mb}MB'")

    idx_dir = config.indexes_dir
    proc_dir = config.processed_dir

    freq_path = idx_dir / "token_frequencies.parquet"
    if not freq_path.exists():
        logger.error(f"Token frequency table {freq_path} not found.")
        return

    df_freq = pd.read_parquet(freq_path)
    rare_tokens_set = set(df_freq[df_freq["doc_freq"] <= 50]["token"].tolist())

    conn.execute(f"CREATE VIEW idx_exact AS SELECT * FROM read_parquet('{idx_dir / 'exact_name_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_core AS SELECT * FROM read_parquet('{idx_dir / 'core_name_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_rare_name AS SELECT * FROM read_parquet('{idx_dir / 'rare_name_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_rare_addr AS SELECT * FROM read_parquet('{idx_dir / 'rare_address_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_numeric AS SELECT * FROM read_parquet('{idx_dir / 'numeric_token_index.parquet'}')")
    conn.execute(f"CREATE VIEW idx_translit AS SELECT * FROM read_parquet('{idx_dir / 'translit_name_index.parquet'}')")

    policy = BlockingPolicy(max_candidates_per_entity=config.max_candidates_per_entity)

    s1_test_files = sorted(list(proc_dir.glob("test_source1_chunk_*.parquet")))
    s1_test_files = [f for f in s1_test_files if not f.name.startswith("id_map_")]

    stage_name = "generate_test_candidates"
    manifest.mark_stage_started(stage_name)

    for chunk_file in s1_test_files:
        chunk_id = chunk_file.stem
        out_parquet = config.candidates_dir / f"candidates_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed test candidates chunk {chunk_id}")
            continue

        df_s1 = pd.read_parquet(chunk_file)
        all_candidate_rows = []

        for _, row in df_s1.iterrows():
            profile = EvidenceProfile(row.to_dict(), rare_tokens_set)
            views = policy.select_views(profile)

            candidates_map = {}

            if "exact_name" in views and profile.normalized_name:
                matches = conn.execute("SELECT internal_id, entity_id FROM idx_exact WHERE normalized_name = ?", [profile.normalized_name]).fetchall()
                for target_int_id, target_ent_id in matches:
                    if target_int_id != profile.internal_id:
                        candidates_map.setdefault((target_int_id, target_ent_id), []).append("exact_name")

            if "core_name" in views and profile.core_name:
                matches = conn.execute("SELECT internal_id, entity_id FROM idx_core WHERE core_name = ?", [profile.core_name]).fetchall()
                for target_int_id, target_ent_id in matches:
                    if target_int_id != profile.internal_id:
                        candidates_map.setdefault((target_int_id, target_ent_id), []).append("core_name")

            if "rare_name_token" in views and profile.rare_name_tokens:
                for tok in profile.rare_name_tokens:
                    matches = conn.execute("SELECT internal_id, entity_id FROM idx_rare_name WHERE token = ?", [tok]).fetchall()
                    for target_int_id, target_ent_id in matches:
                        if target_int_id != profile.internal_id:
                            candidates_map.setdefault((target_int_id, target_ent_id), []).append("rare_name_token")

            if "rare_address_token" in views and profile.rare_address_tokens:
                for tok in profile.rare_address_tokens:
                    matches = conn.execute("SELECT internal_id, entity_id FROM idx_rare_addr WHERE token = ?", [tok]).fetchall()
                    for target_int_id, target_ent_id in matches:
                        if target_int_id != profile.internal_id:
                            candidates_map.setdefault((target_int_id, target_ent_id), []).append("rare_address_token")

            if "numeric_token" in views and profile.numeric_tokens:
                for num_tok in profile.numeric_tokens:
                    matches = conn.execute("SELECT internal_id, entity_id FROM idx_numeric WHERE numeric_token = ?", [num_tok]).fetchall()
                    for target_int_id, target_ent_id in matches:
                        if target_int_id != profile.internal_id:
                            candidates_map.setdefault((target_int_id, target_ent_id), []).append("numeric_token")

            if "translit_name" in views and profile.translit_name:
                matches = conn.execute("SELECT internal_id, entity_id FROM idx_translit WHERE translit_name = ?", [profile.translit_name]).fetchall()
                for target_int_id, target_ent_id in matches:
                    if target_int_id != profile.internal_id:
                        candidates_map.setdefault((target_int_id, target_ent_id), []).append("translit_name")

            overflow = 1 if len(candidates_map) > config.max_candidates_per_entity else 0
            if overflow:
                candidates_map = dict(list(candidates_map.items())[:config.max_candidates_per_entity])

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
            pd.DataFrame(columns=[
                "s1_internal_id", "s1_entity_id", "target_internal_id", "target_entity_id",
                "blocking_views", "num_blocking_views", "overflow"
            ]).to_parquet(out_parquet, index=False)

        manifest.mark_chunk_completed(stage_name, chunk_id, str(out_parquet), len(df_cands))

    conn.close()
    manifest.mark_stage_completed(stage_name)
    log_disk_usage(config.candidates_dir, "Test Candidates Directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Test Candidates")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    generate_test_candidates(cfg)
