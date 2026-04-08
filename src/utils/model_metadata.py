"""
Helpers for reading training metadata that travels with a saved model.

The runtime uses this to understand what market horizon a model was trained
for, so live safety checks can compare the discovered market interval against
the model's native target horizon instead of assuming every model is 5 minutes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

DEFAULT_TARGET_HORIZON_MINUTES = 5
LEGACY_TRAINING_METADATA_FILENAME = "training_metadata.json"


def canonical_training_metadata_path_for_model(model_path: str) -> str:
    """Return the preferred per-model metadata path."""
    return str(Path(model_path).with_suffix(".metadata.json"))


def legacy_training_metadata_path_for_model(model_path: str) -> str:
    """Return the legacy shared metadata path used by older single-model runs."""
    return str(Path(model_path).with_name(LEGACY_TRAINING_METADATA_FILENAME))


def training_metadata_candidate_paths_for_model(model_path: str) -> list[Path]:
    """Return metadata paths in preferred lookup order."""
    canonical = Path(canonical_training_metadata_path_for_model(model_path))
    legacy = Path(legacy_training_metadata_path_for_model(model_path))
    candidates = [canonical]
    if legacy != canonical:
        candidates.append(legacy)
    return candidates


def training_metadata_path_for_model(model_path: str) -> str:
    """Return the discovered metadata path, or the preferred path when missing."""
    for candidate in training_metadata_candidate_paths_for_model(model_path):
        if candidate.exists():
            return str(candidate)
    return canonical_training_metadata_path_for_model(model_path)


def load_training_metadata(model_path: str) -> dict[str, Any]:
    """Best-effort load of the JSON training metadata for a model."""
    for metadata_path in training_metadata_candidate_paths_for_model(model_path):
        if not metadata_path.exists():
            continue

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            return payload

    return {}


def infer_target_horizon_minutes_from_model_path(
    model_path: str,
    *,
    default: int = 0,
) -> int:
    """Infer the target horizon from filenames like `lgbm_btc_60m.txt`."""
    match = re.search(r"(?:^|_)(\d+)m$", Path(model_path).stem)
    if match is None:
        return default

    try:
        horizon = int(match.group(1))
    except (TypeError, ValueError):
        return default

    return horizon if horizon > 0 else default


def uses_legacy_training_metadata(model_path: str) -> bool:
    """Return True when the loader falls back to the old shared metadata file."""
    canonical = Path(canonical_training_metadata_path_for_model(model_path))
    resolved = Path(training_metadata_path_for_model(model_path))
    return (
        not canonical.exists()
        and resolved.name == LEGACY_TRAINING_METADATA_FILENAME
        and resolved.exists()
    )


def resolve_target_horizon_minutes(
    metadata: Optional[Mapping[str, Any]],
    *,
    default: int = DEFAULT_TARGET_HORIZON_MINUTES,
) -> int:
    """Extract a positive target horizon from model metadata."""
    if not metadata:
        return default

    candidates = [
        metadata.get("target_horizon_minutes"),
        (
            metadata.get("training_parameters", {}).get("target_horizon_minutes")
            if isinstance(metadata.get("training_parameters"), Mapping)
            else None
        ),
        (
            metadata.get("dataset_summary", {}).get("target_horizon_minutes")
            if isinstance(metadata.get("dataset_summary"), Mapping)
            else None
        ),
    ]

    for candidate in candidates:
        try:
            horizon = int(candidate)
        except (TypeError, ValueError):
            continue
        if horizon > 0:
            return horizon

    return default


def get_model_target_horizon_minutes(
    model_path: str,
    *,
    default: int = DEFAULT_TARGET_HORIZON_MINUTES,
) -> int:
    """
    Read the target horizon from a model's metadata with a filename fallback.

    Older runs stored one shared metadata file next to multiple model files.
    When that legacy file disagrees with a horizon encoded in the model
    filename, prefer the filename for safety.
    """
    metadata = load_training_metadata(model_path)
    metadata_horizon = resolve_target_horizon_minutes(metadata, default=0)
    inferred_horizon = infer_target_horizon_minutes_from_model_path(
        model_path,
        default=0,
    )

    if metadata_horizon > 0:
        if (
            uses_legacy_training_metadata(model_path)
            and inferred_horizon > 0
            and metadata_horizon != inferred_horizon
        ):
            return inferred_horizon
        return metadata_horizon

    if inferred_horizon > 0:
        return inferred_horizon

    return default
