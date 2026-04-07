"""
Experiment tracking helpers for reproducible model research.

Training and validation should leave a durable record of:
- parameters and dataset summary
- git/runtime context
- metrics and artifacts
"""

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.settings import PATHS


class ExperimentTracker:
    """Persist a multi-stage experiment record plus copied artifacts."""

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        experiments_dir: Optional[str] = None,
    ):
        self._experiments_dir = Path(experiments_dir or PATHS.experiments_dir)
        self._experiment_id = experiment_id or self._build_experiment_id()
        self._experiment_dir = self._experiments_dir / self._experiment_id
        self._record_path = self._experiment_dir / "experiment.json"
        self._record = self._load_or_initialize_record()

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def experiment_dir(self) -> str:
        return str(self._experiment_dir)

    def start_stage(
        self,
        stage: str,
        *,
        label: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark an experiment stage as running and persist its inputs."""
        stage_record = self._record.setdefault("stages", {}).get(stage, {})
        started_at = stage_record.get("started_at_utc") or _utc_now_iso()
        self._record["status"] = "running"
        self._record["updated_at_utc"] = _utc_now_iso()
        self._record["label"] = label or self._record.get("label") or stage
        self._record.setdefault("git", _git_metadata())
        self._record.setdefault("process", _process_metadata())
        self._record.setdefault("artifacts", [])
        self._record.setdefault("stages", {})[stage] = {
            **stage_record,
            "status": "running",
            "started_at_utc": started_at,
            "updated_at_utc": _utc_now_iso(),
            "parameters": parameters or {},
            "context": context or {},
        }
        self._write_record()

    def complete_stage(
        self,
        stage: str,
        *,
        summary: Optional[dict[str, Any]] = None,
        artifacts: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Mark a stage as completed and attach summaries/artifacts."""
        stage_record = self._ensure_stage(stage)
        stage_record["status"] = "completed"
        stage_record["completed_at_utc"] = _utc_now_iso()
        stage_record["updated_at_utc"] = _utc_now_iso()
        if summary is not None:
            stage_record["summary"] = summary
        if artifacts:
            stage_record.setdefault("artifacts", []).extend(artifacts)
            self._record.setdefault("artifacts", []).extend(artifacts)
        self._record["updated_at_utc"] = _utc_now_iso()
        self._record["status"] = _overall_status(self._record.get("stages", {}))
        self._write_record()

    def fail_stage(self, stage: str, error: str) -> None:
        """Mark a stage as failed with an error message."""
        stage_record = self._ensure_stage(stage)
        stage_record["status"] = "failed"
        stage_record["updated_at_utc"] = _utc_now_iso()
        stage_record["failed_at_utc"] = _utc_now_iso()
        stage_record["error"] = error
        self._record["updated_at_utc"] = _utc_now_iso()
        self._record["status"] = "failed"
        self._record["last_error"] = error
        self._write_record()

    def write_json_artifact(
        self,
        relative_path: str,
        payload: dict[str, Any] | list[Any],
    ) -> dict[str, Any]:
        """Write a JSON artifact inside the experiment directory."""
        artifact_path = self._experiment_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self._artifact_metadata(artifact_path)

    def copy_artifact(
        self,
        source_path: str,
        relative_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Copy an existing file into the experiment directory."""
        source = Path(source_path)
        destination = self._experiment_dir / (relative_path or source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return self._artifact_metadata(destination, source_path=str(source.resolve()))

    @classmethod
    def read_experiment_id_from_metadata(
        cls,
        metadata_path: str,
    ) -> Optional[str]:
        """Extract the originating experiment_id from a training metadata file."""
        try:
            payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except Exception:
            return None
        experiment_id = payload.get("experiment_id")
        return str(experiment_id) if experiment_id else None

    @staticmethod
    def build_dataset_summary(
        df: pd.DataFrame,
        *,
        timestamp_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a compact dataset summary for experiment context."""
        summary: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(map(str, df.columns)),
        }

        if timestamp_column and timestamp_column in df.columns and len(df) > 0:
            summary["timestamp_start"] = str(df[timestamp_column].iloc[0])
            summary["timestamp_end"] = str(df[timestamp_column].iloc[-1])

        if target_column and target_column in df.columns and len(df) > 0:
            summary["target_mean"] = float(df[target_column].mean())

        return summary

    @staticmethod
    def summarize_fold_metrics(
        fold_metrics: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Summarize fold metrics into mean/std pairs."""
        if not fold_metrics:
            return {}

        metrics_df = pd.DataFrame(fold_metrics)
        summary: dict[str, dict[str, float]] = {}
        for column in metrics_df.columns:
            if column in {"fold", "train_size", "val_size"}:
                continue
            if not pd.api.types.is_numeric_dtype(metrics_df[column]):
                continue
            summary[column] = {
                "mean": float(metrics_df[column].mean()),
                "std": float(metrics_df[column].std(ddof=1))
                if len(metrics_df[column]) > 1
                else 0.0,
            }
        return summary

    def _load_or_initialize_record(self) -> dict[str, Any]:
        if self._record_path.exists():
            try:
                return json.loads(self._record_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "experiment_id": self._experiment_id,
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
            "status": "created",
            "stages": {},
            "artifacts": [],
        }

    def _ensure_stage(self, stage: str) -> dict[str, Any]:
        stages = self._record.setdefault("stages", {})
        if stage not in stages:
            stages[stage] = {
                "status": "created",
                "started_at_utc": _utc_now_iso(),
                "updated_at_utc": _utc_now_iso(),
            }
        return stages[stage]

    def _write_record(self) -> None:
        self._experiment_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._record_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self._record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._record_path)

    def _artifact_metadata(
        self,
        path: Path,
        *,
        source_path: Optional[str] = None,
    ) -> dict[str, Any]:
        metadata = {
            "path": str(path),
            "relative_path": str(path.relative_to(self._experiment_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if source_path:
            metadata["source_path"] = source_path
        return metadata

    @staticmethod
    def _build_experiment_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"exp-{timestamp}-{uuid.uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overall_status(stages: dict[str, dict[str, Any]]) -> str:
    if any(stage.get("status") == "failed" for stage in stages.values()):
        return "failed"
    if stages and all(stage.get("status") == "completed" for stage in stages.values()):
        return "completed"
    if any(stage.get("status") == "running" for stage in stages.values()):
        return "running"
    return "created"


def _git_metadata() -> dict[str, Optional[str] | bool]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "branch": _safe_git_output(["git", "branch", "--show-current"], cwd=repo_root),
        "commit": _safe_git_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "is_dirty": bool(
            _safe_git_output(["git", "status", "--porcelain"], cwd=repo_root)
        ),
    }


def _safe_git_output(command: list[str], *, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    output = completed.stdout.strip()
    return output or None


def _process_metadata() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "started_at_monotonic": time.monotonic(),
    }
