"""
Unit tests for core utility functions (memory, manifest, text processing, transliteration).
"""

import tempfile
from pathlib import Path
import pytest

from config.default_config import Config
from utils.memory import get_directory_size, get_rss_mb, log_memory
from utils.manifest import ManifestManager
from utils.text_processing import (
    detect_script,
    extract_core_name,
    extract_numeric_tokens,
    normalize_text,
    tokenize_text,
)
from utils.transliteration import transliterate_text


def test_config():
    cfg = Config(project_root="/tmp/test_project", chunk_size=1000)
    assert cfg.chunk_size == 1000
    assert cfg.memory_warning_mb == 4000
    assert cfg.memory_ceiling_mb == 8000
    cfg.ensure_directories()
    assert cfg.processed_dir.exists()


def test_memory_utils():
    rss = get_rss_mb()
    assert rss > 0
    logged_rss = log_memory("test_stage", warning_mb=10000)
    assert logged_rss > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sample.txt"
        test_file.write_text("Hello World!" * 100)
        size_mb, count = get_directory_size(tmpdir)
        assert count == 1
        assert size_mb > 0


def test_manifest_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        mpath = Path(tmpdir) / "manifest.json"
        manifest = ManifestManager(mpath)

        assert not manifest.is_chunk_completed("stage1", "chunk_0")

        dummy_out = Path(tmpdir) / "out.parquet"
        dummy_out.write_text("dummy")

        manifest.mark_chunk_completed("stage1", "chunk_0", str(dummy_out), row_count=100)
        assert manifest.is_chunk_completed("stage1", "chunk_0")
        assert manifest.get_completed_chunks("stage1") == ["chunk_0"]

        # Test reload
        manifest2 = ManifestManager(mpath)
        assert manifest2.is_chunk_completed("stage1", "chunk_0")


def test_text_processing():
    assert normalize_text("  ACME Trading Corp., LTD! ") == "acme trading corp ltd"
    assert extract_core_name("raj investments pvt ltd") == "raj investments"
    assert tokenize_text("Global-Trading Co.") == ["global", "trading", "co"]
    assert extract_numeric_tokens("Door 12-34 Street 5") == ["12", "34", "5"]

    assert detect_script("Acme Corp") == "Latin"
    assert detect_script("राज इन्वेस्टमेंट्स") == "Devanagari"
    assert detect_script("ಗೋಲ್ಡ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್") == "Kannada"
    assert detect_script("ஸ்டார் ஹரைஸன்") == "Tamil"


def test_transliteration():
    # Should handle non-Latin without error
    devanagari = "राज इन्वेस्टमेंट्स"
    res = transliterate_text(devanagari, source_script="Devanagari", enabled=True)
    assert isinstance(res, str)
    assert len(res) > 0

    # Latin should return unchanged
    latin = "Acme Corp"
    assert transliterate_text(latin, source_script="Latin", enabled=True) == latin
