"""Tests for rolling state behavior."""

import numpy as np
import pytest

from src.utils.state import RollingState


class TestRollingState:
    def test_get_window_uses_latest_trade_timestamp(self):
        state = RollingState(maxlen=10)
        for idx, timestamp in enumerate((1000, 4000, 7000), start=1):
            state.push_trade_sync(
                {
                    "price": 100.0 + idx,
                    "quantity": 1.0,
                    "timestamp": timestamp,
                    "is_buyer_maker": False,
                    "trade_id": idx,
                }
            )

        window = state.get_window_by_time(3)

        assert [trade.trade_id for trade in window] == [2, 3]

    def test_not_ready_without_required_history_span(self):
        state = RollingState(maxlen=2000)
        base_timestamp = 1_000_000

        for idx in range(1000):
            state.push_trade_sync(
                {
                    "price": 100.0 + idx * 0.01,
                    "quantity": 1.0,
                    "timestamp": base_timestamp + (idx * 10),
                    "is_buyer_maker": False,
                    "trade_id": idx,
                }
            )

        assert state.size == 1000
        assert state.history_span_seconds < 3600
        assert not state.is_ready

    def test_ready_with_enough_trades_and_history(self):
        state = RollingState(maxlen=2000)
        base_timestamp = 1_000_000

        for idx in range(1000):
            state.push_trade_sync(
                {
                    "price": 100.0 + idx * 0.01,
                    "quantity": 1.0,
                    "timestamp": base_timestamp + (idx * 4000),
                    "is_buyer_maker": False,
                    "trade_id": idx,
                }
            )

        assert state.size == 1000
        assert state.history_span_seconds >= 3600
        assert state.is_ready

    def test_get_price_at_or_before_returns_latest_matching_trade(self):
        state = RollingState(maxlen=10)
        for idx, timestamp in enumerate((1000, 4000, 7000), start=1):
            state.push_trade_sync(
                {
                    "price": 100.0 + idx,
                    "quantity": 1.0,
                    "timestamp": timestamp,
                    "is_buyer_maker": False,
                    "trade_id": idx,
                }
            )

        assert state.get_price_at_or_before(6500) == 102.0
        assert state.get_price_at_or_before(900) is None

    def test_get_volatility_uses_one_second_close_prices(self):
        state = RollingState(maxlen=50)
        base_timestamp = 1_000_000
        closes: list[float] = []
        trade_id = 0

        for second in range(5):
            close_price = 100.0 + (second * 0.02)
            closes.append(close_price)
            for offset, price in (
                (0, close_price + 0.08),
                (200, close_price - 0.08),
                (400, close_price + 0.05),
                (800, close_price),
            ):
                state.push_trade_sync(
                    {
                        "price": price,
                        "quantity": 1.0,
                        "timestamp": base_timestamp + (second * 1000) + offset,
                        "is_buyer_maker": False,
                        "trade_id": trade_id,
                    }
                )
                trade_id += 1

        close_prices = np.array(closes, dtype=np.float64)
        expected = float(np.std(np.log(close_prices[1:] / close_prices[:-1])))

        assert state.get_volatility(5) == pytest.approx(expected)
