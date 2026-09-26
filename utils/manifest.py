"""
Manifest manager for atomic stage checkpointing and pipeline resumability.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ManifestManager:
    """Manages stage checkpoints in manifest.json to allow robust pipeline resume."""

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        self.data: Dict[str, Any] = {"stages": {}, "version": "1.0"}
        self._load()

    def _load(self) -> None:
        """Load manifest from disk if it exists."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading manifest from {self.manifest_path}: {e}. Creating fresh manifest.")
                self.data = {"stages": {}, "version": "1.0"}

    def save(self) -> None:
        """Atomically save manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(temp_path, self.manifest_path)

    def is_chunk_completed(self, stage_name: str, chunk_id: str) -> bool:
        """Check if a specific chunk in a stage is already completed."""
        stage_chunks = self.data.get("stages", {}).get(stage_name, {}).get("chunks", {})
        chunk_entry = stage_chunks.get(chunk_id)
        if chunk_entry and chunk_entry.get("status") == "completed":
            # Verify output file exists
            out_path = chunk_entry.get("output_path")
            if out_path and Path(out_path).exists():
                return True
        return False

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if an entire stage is marked completed."""
        stage_data = self.data.get("stages", {}).get(stage_name, {})
        return stage_data.get("status") == "completed"

    def mark_stage_started(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a stage as started."""
        if "stages" not in self.data:
            self.data["stages"] = {}
        if stage_name not in self.data["stages"]:
            self.data["stages"][stage_name] = {"chunks": {}, "status": "in_progress"}

        self.data["stages"][stage_name]["status"] = "in_progress"
        self.data["stages"][stage_name]["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if metadata:
            self.data["stages"][stage_name].update(metadata)
        self.save()

    def mark_chunk_completed(
        self,
        stage_name: str,
        chunk_id: str,
        output_path: str,
        row_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record chunk completion in manifest."""
        if stage_name not in self.data["stages"]:
            self.mark_stage_started(stage_name)

        chunk_record = {
            "status": "completed",
            "output_path": str(output_path),
            "row_count": row_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            chunk_record["metadata"] = metadata

        self.data["stages"][stage_name]["chunks"][chunk_id] = chunk_record
        self.save()

    def mark_chunk_failed(
        self,
        stage_name: str,
        chunk_id: str,
        error_msg: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record chunk failure in manifest."""
        if stage_name not in self.data["stages"]:
            self.mark_stage_started(stage_name)

        chunk_record = {
            "status": "failed",
            "error": error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metadata:
            chunk_record["metadata"] = metadata

        self.data["stages"][stage_name]["chunks"][chunk_id] = chunk_record
        self.save()

    def mark_stage_completed(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark stage as fully completed."""
        if stage_name not in self.data["stages"]:
            self.mark_stage_started(stage_name)

        self.data["stages"][stage_name]["status"] = "completed"
        self.data["stages"][stage_name]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if metadata:
            self.data["stages"][stage_name].update(metadata)
        self.save()

    def get_completed_chunks(self, stage_name: str) -> List[str]:
        """Get list of completed chunk IDs for a stage."""
        stage_chunks = self.data.get("stages", {}).get(stage_name, {}).get("chunks", {})
        return [cid for cid, info in stage_chunks.items() if info.get("status") == "completed"]
