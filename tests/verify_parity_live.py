import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.pipeline import FeaturePipeline
from src.features.schema import FEATURE_COLUMNS
from src.utils.state import RollingState


spec = spec_from_file_location(
    "engineer_features",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "02_engineer_features.py",
    ),
)
engineer = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(engineer)


def verify_parity():
    print("[*] Running full feature parity verification...")

    state = RollingState(maxlen=20000)
    rows = []
    base_ts = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)
    price = 65_000.0
    trade_id = 0
    rng = np.random.default_rng(42)

    for minute in range(120):
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

    offline = engineer.compute_features(pd.DataFrame(rows))
    expected = offline.iloc[-1][list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    actual = FeaturePipeline(state).compute()

    if actual is None:
        raise AssertionError("Live pipeline did not emit a feature vector")

    diffs = np.abs(actual - expected)
    max_diff = float(np.max(diffs))

    for name, expected_value, actual_value, diff in zip(
        FEATURE_COLUMNS, expected, actual, diffs
    ):
        print(
            f"[*] {name:18s} | expected={expected_value:.10f} "
            f"actual={actual_value:.10f} diff={diff:.3e}"
        )

    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    print(f"\n[✓] Full parity verified across {len(FEATURE_COLUMNS)} features.")
    print(f"[✓] Maximum absolute diff: {max_diff:.3e}")


if __name__ == "__main__":
    verify_parity()
