from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.features.schema import FEATURE_COLUMNS
from src.utils.state import RollingState


spec = spec_from_file_location(
    "engineer_features",
    "scripts/02_engineer_features.py",
)
engineer = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(engineer)


def _build_state_and_bars(minutes: int = 120, extra_partial_seconds: int = 0):
    state = RollingState(maxlen=20000)
    rows: list[dict] = []
    base_ts = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)
    price = 65_000.0
    trade_id = 0
    rng = np.random.default_rng(42)

    for minute in range(minutes):
        open_price = price
        tick_prices = []
        tick_qty = []
        tick_bm = []

        for second in range(60):
            step = (0.5 if minute % 2 == 0 else -0.3) + rng.normal(0, 0.2)
            price = max(1.0, price + step)
            quantity = float(rng.uniform(0.5, 2.0))
            is_buyer_maker = bool(rng.integers(0, 2))
            timestamp = base_ts + (minute * 60 + second) * 1000

            tick_prices.append(price)
            tick_qty.append(quantity)
            tick_bm.append(is_buyer_maker)
            state.push_trade_sync(
                {
                    "price": price,
                    "quantity": quantity,
                    "timestamp": timestamp,
                    "is_buyer_maker": is_buyer_maker,
                    "trade_id": trade_id,
                }
            )
            trade_id += 1

        rows.append(
            {
                "open_time": pd.Timestamp(base_ts + minute * 60_000, unit="ms"),
                "close_time": pd.Timestamp(
                    base_ts + (minute + 1) * 60_000 - 1,
                    unit="ms",
                ),
                "open": open_price,
                "high": max(tick_prices),
                "low": min(tick_prices),
                "close": tick_prices[-1],
                "volume": sum(tick_qty),
                "taker_buy_base": sum(
                    quantity
                    for quantity, is_buyer_maker in zip(tick_qty, tick_bm)
                    if not is_buyer_maker
                ),
                "trades_count": 60,
            }
        )

    if extra_partial_seconds:
        for second in range(extra_partial_seconds):
            price += 25.0
            timestamp = base_ts + (minutes * 60 + second) * 1000
            state.push_trade_sync(
                {
                    "price": price,
                    "quantity": 5.0,
                    "timestamp": timestamp,
                    "is_buyer_maker": False,
                    "trade_id": trade_id,
                }
            )
            trade_id += 1

    return state, pd.DataFrame(rows)


def test_live_pipeline_matches_offline_training_features():
    state, bars = _build_state_and_bars()

    offline = engineer.compute_features(bars.copy())
    expected = offline.iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    live = FeaturePipeline(state).compute()

    assert live is not None
    np.testing.assert_allclose(live, expected, rtol=1e-9, atol=1e-9)


def test_live_pipeline_can_run_from_seeded_historical_bars_only():
    state, bars = _build_state_and_bars()
    pipeline = FeaturePipeline(state)
    pipeline.seed_historical_bars(bars.copy())

    offline = engineer.compute_features(bars.copy())
    expected = offline.iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    live = pipeline.compute()

    assert pipeline.is_ready is True
    assert live is not None
    np.testing.assert_allclose(live, expected, rtol=1e-9, atol=1e-9)


def test_live_pipeline_ignores_incomplete_last_minute():
    state, bars = _build_state_and_bars(extra_partial_seconds=30)

    offline = engineer.compute_features(bars.copy())
    expected = offline.iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    live = FeaturePipeline(state).compute()

    assert live is not None
    np.testing.assert_allclose(live, expected, rtol=1e-9, atol=1e-9)
