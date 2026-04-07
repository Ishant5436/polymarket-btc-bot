"""Tests for shared feature schema consistency."""

from src.features.pipeline import FeaturePipeline
from src.features.schema import FEATURE_COLUMNS
from src.utils.state import RollingState


def test_live_pipeline_uses_shared_feature_schema():
    pipeline = FeaturePipeline(RollingState())

    assert pipeline.feature_names == list(FEATURE_COLUMNS)
    assert pipeline.feature_count == len(FEATURE_COLUMNS)
