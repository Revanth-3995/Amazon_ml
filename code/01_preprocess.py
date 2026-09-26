"""
Chunked Preprocessing Module.
Reads raw TSV files in chunks, normalizes names/addresses, extracts tokens, assigns compact integer internal IDs,
maintains disk-backed ID mappings, and outputs partitioned Parquet chunks with stage checkpointing.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.default_config import Config
from utils.manifest import ManifestManager
from utils.memory import get_rss_mb, log_disk_usage, log_memory
from utils.text_processing import (
    detect_script,
    extract_core_name,
    extract_numeric_tokens,
    normalize_text,
    tokenize_text,
)
from utils.transliteration import transliterate_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("01_preprocess")


def preprocess_source_file(
    file_path: Path,
    source_name: str,
    output_dir: Path,
    chunk_size: int,
    manifest: ManifestManager,
    start_internal_id: int = 0,
    enable_transliteration: bool = True,
    resume: bool = True,
) -> Tuple[int, int]:
    """
    Process a single TSV source file in chunks.
    Writes partitioned Parquet chunks to disk and updates ID mappings.
    Returns: (total_processed_rows, next_start_internal_id)
    """
    if not file_path.exists():
        logger.warning(f"Source file {file_path} does not exist. Skipping.")
        return 0, start_internal_id

    stage_name = f"preprocess_{source_name}"
    manifest.mark_stage_started(stage_name)

    logger.info(f"Starting chunked preprocessing for {source_name} from {file_path}")

    total_processed_rows = 0
    chunk_idx = 0
    id_counter = start_internal_id

    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
    )

    for df_chunk in reader:
        chunk_id = f"chunk_{chunk_idx:04d}"
        output_file = output_dir / f"{source_name}_{chunk_id}.parquet"

        if resume and manifest.is_chunk_completed(stage_name, chunk_id):
            logger.info(f"Skipping completed chunk {chunk_id} for {source_name}")
            id_counter += len(df_chunk)
            total_processed_rows += len(df_chunk)
            chunk_idx += 1
            continue

        processed_records = []
        id_mappings = []

        for _, row in df_chunk.iterrows():
            orig_entity_id = str(row.get("entity_id", "")).strip()
            internal_id = id_counter
            id_counter += 1

            orig_name = str(row.get("business_name", "")).strip()
            orig_addr = str(row.get("business_address", "")).strip()
            orig_country = str(row.get("country", "")).strip().upper()

            norm_name = normalize_text(orig_name)
            core_name = extract_core_name(norm_name)
            name_tokens = tokenize_text(norm_name)

            norm_addr = normalize_text(orig_addr) if orig_addr else ""
            addr_tokens = tokenize_text(norm_addr) if norm_addr else []
            num_tokens = extract_numeric_tokens(orig_name + " " + orig_addr)

            name_script = detect_script(orig_name)
            addr_script = detect_script(orig_addr) if orig_addr else "None"

            # Optional transliteration
            translit_name = transliterate_text(orig_name, source_script=name_script, enabled=enable_transliteration)
            translit_norm_name = normalize_text(translit_name) if translit_name != orig_name else norm_name

            processed_records.append({
                "internal_id": internal_id,
                "entity_id": orig_entity_id,
                "source_name": source_name,
                "original_name": orig_name,
                "normalized_name": norm_name,
                "core_name": core_name,
                "name_tokens": " ".join(name_tokens),
                "original_address": orig_addr,
                "normalized_address": norm_addr,
                "address_tokens": " ".join(addr_tokens),
                "numeric_tokens": " ".join(num_tokens),
                "country": orig_country,
                "name_script": name_script,
                "address_script": addr_script,
                "translit_name": translit_norm_name,
                "name_missing": 1 if not norm_name else 0,
                "address_missing": 1 if not norm_addr else 0,
            })

            id_mappings.append({
                "internal_id": internal_id,
                "original_entity_id": orig_entity_id,
                "source_name": source_name,
            })

        df_proc = pd.DataFrame(processed_records)
        df_proc.to_parquet(output_file, index=False)

        # Write chunk ID map
        id_map_file = output_dir / f"id_map_{source_name}_{chunk_id}.parquet"
        pd.DataFrame(id_mappings).to_parquet(id_map_file, index=False)

        chunk_rows = len(df_proc)
        total_processed_rows += chunk_rows

        manifest.mark_chunk_completed(stage_name, chunk_id, str(output_file), chunk_rows)

        log_memory(f"Preprocess {source_name} {chunk_id}")
        chunk_idx += 1

    manifest.mark_stage_completed(stage_name, {"total_rows": total_processed_rows, "end_internal_id": id_counter})
    logger.info(f"Finished preprocessing {source_name}: {total_processed_rows} total rows.")
    return total_processed_rows, id_counter


def run_preprocessing(config: Config, resume: bool = True) -> None:
    """Run preprocessing across all train and test sources with globally unique internal IDs."""
    config.ensure_directories()
    manifest = ManifestManager(config.processed_dir / "manifest.json")

    sources = [
        (config.train_dataset_dir / "train_source1.tsv", "train_source1"),
        (config.train_dataset_dir / "train_source2.tsv", "train_source2"),
        (config.train_dataset_dir / "train_source3.tsv", "train_source3"),
        (config.test_dataset_dir / "test_source1.tsv", "test_source1"),
        (config.test_dataset_dir / "test_source2.tsv", "test_source2"),
        (config.test_dataset_dir / "test_source3.tsv", "test_source3"),
    ]

    current_id = 0
    for file_path, source_name in sources:
        rows, next_id = preprocess_source_file(
            file_path=file_path,
            source_name=source_name,
            output_dir=config.processed_dir,
            chunk_size=config.chunk_size,
            manifest=manifest,
            start_internal_id=current_id,
            enable_transliteration=config.enable_transliteration,
            resume=resume,
        )
        current_id = next_id

    log_disk_usage(config.processed_dir, "Processed Directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked Preprocessing for Business Entity Resolution")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for reading TSVs")
    parser.add_argument("--no-resume", action="store_true", help="Force restart preprocessing stage")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root, chunk_size=args.chunk_size)
    run_preprocessing(cfg, resume=not args.no_resume)
