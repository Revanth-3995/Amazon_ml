# Memory & RAM Design Specifications

## Colab Constraints & Budget Targets
- **System Total RAM**: ~12.7 GB (Colab CPU environment)
- **Target Normal RSS**: < 2 GB
- **Preferred Peak RSS**: < 4 GB
- **Design Ceiling**: < 8 GB

## Memory Safety Principles
1. **No In-Memory Monoliths**: Never execute `pd.read_csv()` or `pd.read_parquet()` on full multi-million row datasets.
2. **Chunked Streaming**: Process source data in configurable chunks (default: `chunk_size = 50,000`).
3. **DuckDB Direct File Querying**: Use DuckDB SQL queries that stream directly to and from Parquet files without materializing full tables in pandas DataFrames.
4. **Data Type Downcasting**:
   - Scores & Similarities: `float32`
   - Binary Indicators & Labels: `int8`
   - Counts & Views: `int16`
   - Internal Entity IDs: `int32`
5. **Active Garbage Collection & Monitoring**: `utils/memory.py` monitors RSS continuously (`get_rss_mb()`) and automatically invokes `gc.collect()` when RSS crosses threshold (`4000 MB`).
