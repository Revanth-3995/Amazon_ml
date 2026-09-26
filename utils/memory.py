"""
Memory and Disk footprint tracking utilities for Colab safety.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    """Get process Resident Set Size (RSS) in Megabytes."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    else:
        # Fallback to /proc/self/statm if on Linux without psutil
        try:
            with open("/proc/self/statm", "r") as f:
                pages = int(f.read().split()[1])
                return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024 * 1024)
        except Exception:
            return 0.0


def get_available_memory_mb() -> float:
    """Get available system RAM in Megabytes."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 * 1024)
    else:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def log_memory(stage: str, warning_mb: float = 4000.0) -> float:
    """
    Log memory status for a given stage.
    Triggers garbage collection if RSS exceeds warning threshold.
    Returns current RSS in MB.
    """
    rss = get_rss_mb()
    avail = get_available_memory_mb()
    msg = f"[Memory Log - {stage}] RSS: {rss:.2f} MB | Avail System RAM: {avail:.2f} MB"

    if rss > warning_mb:
        logger.warning(
            f"MEMORY WARNING: RSS {rss:.2f} MB exceeds threshold {warning_mb:.2f} MB. Running gc.collect()."
        )
        gc.collect()
        rss_after = get_rss_mb()
        logger.info(f"Memory after gc.collect(): {rss_after:.2f} MB")
        return rss_after
    else:
        logger.info(msg)
        return rss


def get_directory_size(dir_path: Union[str, Path]) -> Tuple[float, int]:
    """
    Calculate directory size in Megabytes and return total file count.
    Returns: (size_in_mb, file_count)
    """
    path = Path(dir_path)
    if not path.exists():
        return 0.0, 0

    total_size_bytes = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for f in files:
            file_path = Path(root) / f
            if file_path.is_file() and not file_path.is_symlink():
                total_size_bytes += file_path.stat().st_size
                file_count += 1

    size_mb = total_size_bytes / (1024 * 1024)
    return size_mb, file_count


def log_disk_usage(dir_path: Union[str, Path], label: str = "") -> Dict[str, Union[float, int]]:
    """Log size and file count of a directory."""
    size_mb, file_count = get_directory_size(dir_path)
    logger.info(f"[Disk Usage - {label or str(dir_path)}] Size: {size_mb:.2f} MB across {file_count} files")
    return {"size_mb": size_mb, "file_count": file_count}
