"""
Helpers for reading training metadata that travels with a saved model.

The runtime uses this to understand what market horizon a model was trained
for, so live safety checks can compare the discovered market interval against
the model's native target horizon instead of assuming every model is 5 minutes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

DEFAULT_TARGET_HORIZON_MINUTES = 5


def training_metadata_path_for_model(model_path: str) -> str:
    """Return the companion metadata path for a serialized model."""
    return str(Path(model_path).with_name("training_metadata.json"))


def load_training_metadata(model_path: str) -> dict[str, Any]:
    """Best-effort load of the JSON training metadata for a model."""
    metadata_path = Path(training_metadata_path_for_model(model_path))
    if not metadata_path.exists():
        return {}

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


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
    """Read the target horizon from a model's companion metadata."""
    return resolve_target_horizon_minutes(
        load_training_metadata(model_path),
        default=default,
    )
