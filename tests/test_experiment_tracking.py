import json
from pathlib import Path

import pandas as pd

from src.utils.experiment_tracking import ExperimentTracker


def test_build_dataset_summary_includes_time_range_and_target_mean():
    df = pd.DataFrame(
        {
            "open_time": ["2026-04-05T00:00:00Z", "2026-04-05T00:01:00Z"],
            "feature_a": [1.0, 2.0],
            "target": [0, 1],
        }
    )

    summary = ExperimentTracker.build_dataset_summary(
        df,
        timestamp_column="open_time",
        target_column="target",
    )

    assert summary["rows"] == 2
    assert summary["columns"] == 3
    assert summary["timestamp_start"] == "2026-04-05T00:00:00Z"
    assert summary["timestamp_end"] == "2026-04-05T00:01:00Z"
    assert summary["target_mean"] == 0.5


def test_summarize_fold_metrics_returns_mean_and_std():
    summary = ExperimentTracker.summarize_fold_metrics(
        [
            {"fold": 1, "auc": 0.6, "accuracy": 0.55, "best_iteration": 10},
            {"fold": 2, "auc": 0.8, "accuracy": 0.65, "best_iteration": 14},
        ]
    )

    assert summary["auc"]["mean"] == 0.7
    assert round(summary["auc"]["std"], 6) == round((0.02**0.5), 6)
    assert summary["best_iteration"]["mean"] == 12.0


def test_tracker_persists_multistage_record_and_artifacts(tmp_path):
    tracker = ExperimentTracker(
        experiment_id="exp-20260406T120000Z-test1234",
        experiments_dir=str(tmp_path),
    )
    source_path = tmp_path / "source.txt"
    source_path.write_text("artifact-body", encoding="utf-8")

    tracker.start_stage(
        "training",
        label="train_model",
        parameters={"n_splits": 5},
        context={"rows": 100},
    )
    json_artifact = tracker.write_json_artifact(
        "reports/training.json",
        {"auc": 0.7},
    )
    copied_artifact = tracker.copy_artifact(
        str(source_path),
        "model/source.txt",
    )
    tracker.complete_stage(
        "training",
        summary={"auc": 0.7},
        artifacts=[json_artifact, copied_artifact],
    )

    resumed = ExperimentTracker(
        experiment_id="exp-20260406T120000Z-test1234",
        experiments_dir=str(tmp_path),
    )
    resumed.start_stage("validation", parameters={"holdout_pct": 0.2})
    resumed.complete_stage("validation", summary={"accuracy": 0.6})

    record_path = Path(tmp_path) / "exp-20260406T120000Z-test1234" / "experiment.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["status"] == "completed"
    assert set(record["stages"].keys()) == {"training", "validation"}
    assert record["stages"]["training"]["summary"]["auc"] == 0.7
    assert record["stages"]["validation"]["summary"]["accuracy"] == 0.6
    assert len(record["artifacts"]) == 2


def test_read_experiment_id_from_metadata(tmp_path):
    metadata_path = tmp_path / "training_metadata.json"
    metadata_path.write_text(
        json.dumps({"experiment_id": "exp-abc123"}),
        encoding="utf-8",
    )

    assert (
        ExperimentTracker.read_experiment_id_from_metadata(str(metadata_path))
        == "exp-abc123"
    )
