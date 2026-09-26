"""
Disk-Backed Blocking Strategy Benchmark.
Evaluates 10 individual and combined blocking strategies on true pairs in train_ground_truth.tsv.
Computes recall, candidate volume statistics (mean, median, P95, P99), runtime, and peak RSS memory.
Outputs JSON, CSV, and Markdown report without materializing giant candidate sets in memory.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import duckdb
import numpy as np
import pandas as pd

from config.default_config import Config
from utils.memory import get_rss_mb, log_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("03_blocking_benchmark")


def run_blocking_benchmark(config: Config) -> Dict[str, Any]:
    """Execute blocking benchmark using streaming DuckDB queries."""
    config.ensure_directories()

    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET max_memory='{config.memory_warning_mb}MB'")

    gt_file = config.train_dataset_dir / "train_ground_truth.tsv"
    if not gt_file.exists():
        logger.error(f"Ground truth file {gt_file} not found. Cannot run benchmark.")
        return {}

    conn.execute(f"""
        CREATE TABLE ground_truth AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ' ')) AS target_entity_id
        FROM read_csv_auto('{gt_file}', delim='\t', header=True)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) != ''
    """)

    total_gt_pairs = conn.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
    total_s1_gt = conn.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM ground_truth").fetchone()[0]
    logger.info(f"Loaded ground truth: {total_gt_pairs} true pairs across {total_s1_gt} S1 entities.")

    parquet_files = [
        f for f in config.processed_dir.glob("*.parquet")
        if not f.name.startswith("id_map_") and not f.name.startswith("training_pairs_")
    ]
    if not parquet_files:
        logger.error("No processed parquet files found. Run 01_preprocess.py first.")
        return {}

    file_list_str = ", ".join([f"'{str(f)}'" for f in parquet_files])
    conn.execute(f"CREATE VIEW processed_all AS SELECT * FROM read_parquet([{file_list_str}])")

    # Create indexes views
    idx_dir = config.indexes_dir
    conn.execute(f"CREATE VIEW idx_exact AS SELECT * FROM read_parquet('{idx_dir / 'exact_name_index.parquet'}') WHERE source_name IN ('train_source2', 'train_source3')")
    conn.execute(f"CREATE VIEW idx_core AS SELECT * FROM read_parquet('{idx_dir / 'core_name_index.parquet'}') WHERE source_name IN ('train_source2', 'train_source3')")
    conn.execute(f"CREATE VIEW idx_rare_name AS SELECT * FROM read_parquet('{idx_dir / 'rare_name_token_index.parquet'}') WHERE source_name IN ('train_source2', 'train_source3')")
    conn.execute(f"CREATE VIEW idx_rare_addr AS SELECT * FROM read_parquet('{idx_dir / 'rare_address_token_index.parquet'}') WHERE source_name IN ('train_source2', 'train_source3')")
    conn.execute(f"CREATE VIEW idx_numeric AS SELECT * FROM read_parquet('{idx_dir / 'numeric_token_index.parquet'}') WHERE source_name IN ('train_source2', 'train_source3')")

    strategies = [
        ("country", """
            SELECT p.entity_id AS s1_id, t.entity_id AS target_id
            FROM processed_all p JOIN processed_all t ON p.country = t.country AND p.internal_id != t.internal_id
            WHERE p.source_name = 'train_source1' AND t.source_name IN ('train_source2', 'train_source3')
        """),
        ("exact_name", """
            SELECT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM processed_all p JOIN idx_exact idx ON p.normalized_name = idx.normalized_name AND p.internal_id != idx.internal_id
            WHERE p.source_name = 'train_source1' AND p.normalized_name != ''
        """),
        ("core_name", """
            SELECT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM processed_all p JOIN idx_core idx ON p.core_name = idx.core_name AND p.internal_id != idx.internal_id
            WHERE p.source_name = 'train_source1' AND p.core_name != ''
        """),
        ("rare_name_token", """
            SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM (
                SELECT entity_id, unnest(string_split(name_tokens, ' ')) AS token
                FROM processed_all WHERE source_name = 'train_source1' AND name_tokens != ''
            ) p JOIN idx_rare_name idx ON p.token = idx.token AND p.entity_id != idx.entity_id
        """),
        ("rare_address_token", """
            SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM (
                SELECT entity_id, unnest(string_split(address_tokens, ' ')) AS token
                FROM processed_all WHERE source_name = 'train_source1' AND address_tokens != ''
            ) p JOIN idx_rare_addr idx ON p.token = idx.token AND p.entity_id != idx.entity_id
        """),
        ("numeric_token", """
            SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM (
                SELECT entity_id, unnest(string_split(numeric_tokens, ' ')) AS numeric_token
                FROM processed_all WHERE source_name = 'train_source1' AND numeric_tokens != ''
            ) p JOIN idx_numeric idx ON p.numeric_token = idx.numeric_token AND p.entity_id != idx.entity_id
        """),
        ("name_and_address", """
            SELECT p.entity_id AS s1_id, t.entity_id AS target_id
            FROM processed_all p JOIN processed_all t
            ON p.core_name = t.core_name AND p.normalized_address = t.normalized_address AND p.internal_id != t.internal_id
            WHERE p.source_name = 'train_source1' AND t.source_name IN ('train_source2', 'train_source3') AND p.core_name != '' AND p.normalized_address != ''
        """),
        ("name_and_numeric", """
            SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM (
                SELECT entity_id, core_name, unnest(string_split(numeric_tokens, ' ')) AS num_tok
                FROM processed_all WHERE source_name = 'train_source1' AND core_name != '' AND numeric_tokens != ''
            ) p JOIN (
                SELECT entity_id, core_name, unnest(string_split(numeric_tokens, ' ')) AS num_tok
                FROM processed_all WHERE source_name IN ('train_source2', 'train_source3') AND core_name != '' AND numeric_tokens != ''
            ) idx ON p.core_name = idx.core_name AND p.num_tok = idx.num_tok AND p.entity_id != idx.entity_id
        """),
        ("address_and_numeric", """
            SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
            FROM (
                SELECT entity_id, normalized_address, unnest(string_split(numeric_tokens, ' ')) AS num_tok
                FROM processed_all WHERE source_name = 'train_source1' AND normalized_address != '' AND numeric_tokens != ''
            ) p JOIN (
                SELECT entity_id, normalized_address, unnest(string_split(numeric_tokens, ' ')) AS num_tok
                FROM processed_all WHERE source_name IN ('train_source2', 'train_source3') AND normalized_address != '' AND numeric_tokens != ''
            ) idx ON p.normalized_address = idx.normalized_address AND p.num_tok = idx.num_tok AND p.entity_id != idx.entity_id
        """),
        ("adaptive_multi_view", """
            SELECT DISTINCT s1_id, target_id FROM (
                SELECT p.entity_id AS s1_id, idx.entity_id AS target_id
                FROM processed_all p JOIN idx_core idx ON p.core_name = idx.core_name AND p.internal_id != idx.internal_id
                WHERE p.source_name = 'train_source1' AND p.core_name != ''
                UNION ALL
                SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
                FROM (
                    SELECT entity_id, unnest(string_split(address_tokens, ' ')) AS token
                    FROM processed_all WHERE source_name = 'train_source1' AND address_tokens != ''
                ) p JOIN idx_rare_addr idx ON p.token = idx.token AND p.entity_id != idx.entity_id
                UNION ALL
                SELECT DISTINCT p.entity_id AS s1_id, idx.entity_id AS target_id
                FROM (
                    SELECT entity_id, unnest(string_split(numeric_tokens, ' ')) AS numeric_token
                    FROM processed_all WHERE source_name = 'train_source1' AND numeric_tokens != ''
                ) p JOIN idx_numeric idx ON p.numeric_token = idx.numeric_token AND p.entity_id != idx.entity_id
            )
        """),
    ]

    results = []

    for name, query in strategies:
        t0 = time.time()
        conn.execute(f"CREATE TEMP TABLE candidates_temp AS {query}")

        matched_gt = conn.execute("""
            SELECT COUNT(*) FROM ground_truth gt
            JOIN candidates_temp c ON gt.source1_entity_id = c.s1_id AND gt.target_entity_id = c.target_id
        """).fetchone()[0]

        recall = matched_gt / total_gt_pairs if total_gt_pairs > 0 else 0.0

        counts_df = conn.execute("""
            SELECT s1_id, COUNT(*) AS cand_count
            FROM candidates_temp GROUP BY s1_id
        """).fetchdf()

        if len(counts_df) > 0:
            counts = counts_df["cand_count"].values
            mean_c = float(np.mean(counts))
            median_c = float(np.median(counts))
            p95_c = float(np.percentile(counts, 95))
            p99_c = float(np.percentile(counts, 99))
            total_c = int(np.sum(counts))
        else:
            mean_c = median_c = p95_c = p99_c = 0.0
            total_c = 0

        elapsed = time.time() - t0
        peak_rss = get_rss_mb()

        conn.execute("DROP TABLE candidates_temp")

        res_entry = {
            "strategy": name,
            "candidate_recall": round(recall, 4),
            "matched_gt_pairs": matched_gt,
            "total_gt_pairs": total_gt_pairs,
            "total_candidates": total_c,
            "mean_candidates_per_s1": round(mean_c, 2),
            "median_candidates_per_s1": round(median_c, 2),
            "p95_candidates_per_s1": round(p95_c, 2),
            "p99_candidates_per_s1": round(p99_c, 2),
            "runtime_seconds": round(elapsed, 2),
            "rss_peak_mb": round(peak_rss, 2),
        }
        results.append(res_entry)
        logger.info(f"Strategy [{name}]: Recall={recall:.4f}, Total Cands={total_c}, Mean/S1={mean_c:.1f}, Time={elapsed:.2f}s")

    conn.close()

    out_dir = config.evaluation_dir
    json_path = out_dir / "blocking_benchmark.json"
    csv_path = out_dir / "blocking_benchmark.csv"
    md_path = out_dir / "blocking_benchmark_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)

    md_lines = [
        "# Blocking Strategy Benchmark Report",
        "",
        "Note: Measured using streaming DuckDB queries on available dataset.",
        "",
        "| Strategy | Candidate Recall | Matched GT | Total Cands | Mean/S1 | Median/S1 | P95/S1 | P99/S1 | Time (s) | Peak RSS (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        md_lines.append(
            f"| {r['strategy']} | {r['candidate_recall']:.4f} | {r['matched_gt_pairs']} | {r['total_candidates']} | "
            f"{r['mean_candidates_per_s1']} | {r['median_candidates_per_s1']} | {r['p95_candidates_per_s1']} | "
            f"{r['p99_candidates_per_s1']} | {r['runtime_seconds']} | {r['rss_peak_mb']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Saved benchmark report to {json_path}, {csv_path}, and {md_path}")
    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocking Benchmark Tool")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path")
    args = parser.parse_args()

    cfg = Config(project_root=args.project_root)
    run_blocking_benchmark(cfg)
