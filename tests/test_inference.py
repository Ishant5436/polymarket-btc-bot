"""Tests for the model inference module."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from src.execution.inference import ModelInference


@pytest.fixture
def dummy_model_path(tmp_path):
    """Create a temporary dummy LightGBM model for testing."""
    # Train a simple model
    np.random.seed(42)
    X = np.random.randn(200, 20)
    y = (X[:, 0] > 0).astype(int)

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "num_leaves": 4,
        "n_jobs": 1,
    }
    model = lgb.train(params, train_data, num_boost_round=10)

    model_path = tmp_path / "dummy_model.txt"
    model.save_model(str(model_path))
    yield str(model_path)


class TestModelInference:
    def test_load_model(self, dummy_model_path):
        inf = ModelInference(model_path=dummy_model_path)
        assert not inf.is_loaded
        assert inf.load()
        assert inf.is_loaded
        assert inf.target_horizon_minutes == 5

    def test_load_model_reads_target_horizon_from_metadata(self, dummy_model_path):
        metadata_path = Path(dummy_model_path).with_name("training_metadata.json")
        metadata_path.write_text(
            json.dumps({"target_horizon_minutes": 60}),
            encoding="utf-8",
        )

        inf = ModelInference(model_path=dummy_model_path)

        assert inf.load()
        assert inf.target_horizon_minutes == 60

    def test_load_nonexistent(self):
        inf = ModelInference(model_path="/nonexistent/path.txt")
        assert not inf.load()

    def test_predict(self, dummy_model_path):
        inf = ModelInference(model_path=dummy_model_path)
        inf.load()

        features = np.random.randn(20)
        prob = inf.predict(features)

        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_predict_without_load(self):
        inf = ModelInference(model_path="/fake")
        result = inf.predict(np.zeros(20))
        assert result is None

    def test_prediction_count(self, dummy_model_path):
        inf = ModelInference(model_path=dummy_model_path)
        inf.load()

        assert inf.prediction_count == 0
        inf.predict(np.random.randn(20))
        assert inf.prediction_count == 1
        inf.predict(np.random.randn(20))
        assert inf.prediction_count == 2

    def test_predict_batch(self, dummy_model_path):
        inf = ModelInference(model_path=dummy_model_path)
        inf.load()

        X = np.random.randn(10, 20)
        probs = inf.predict_batch(X)

        assert probs is not None
        assert len(probs) == 10
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_deterministic_predictions(self, dummy_model_path):
        """Same input should produce same output."""
        inf = ModelInference(model_path=dummy_model_path)
        inf.load()

        features = np.ones(20) * 0.5
        p1 = inf.predict(features)
        p2 = inf.predict(features)

        assert p1 == pytest.approx(p2)
