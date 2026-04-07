"""Tests for feature engineering indicators."""

import numpy as np
import pytest

from src.features.indicators import (
    fractional_differentiation,
    micro_price_momentum,
    order_book_imbalance,
    rolling_hurst_exponent,
    rolling_volatility,
    trade_flow_imbalance,
    vwap_deviation,
)


class TestOrderBookImbalance:
    def test_balanced_book(self):
        bid = np.array([100.0, 100.0, 100.0])
        ask = np.array([100.0, 100.0, 100.0])
        result = order_book_imbalance(bid, ask)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_buy_pressure(self):
        bid = np.array([200.0])
        ask = np.array([100.0])
        result = order_book_imbalance(bid, ask)
        assert result[0] == pytest.approx(1 / 3)

    def test_sell_pressure(self):
        bid = np.array([50.0])
        ask = np.array([150.0])
        result = order_book_imbalance(bid, ask)
        assert result[0] == pytest.approx(-0.5)

    def test_zero_volume(self):
        bid = np.array([0.0])
        ask = np.array([0.0])
        result = order_book_imbalance(bid, ask)
        assert result[0] == 0.0


class TestMicroPriceMomentum:
    def test_upward_momentum(self):
        prices = np.array([100.0, 102.0, 104.0, 106.0])
        result = micro_price_momentum(prices, window=1)
        assert np.isnan(result[0])
        assert result[1] > 0  # Price went up

    def test_downward_momentum(self):
        prices = np.array([100.0, 98.0, 96.0])
        result = micro_price_momentum(prices, window=1)
        assert result[1] < 0  # Price went down

    def test_window_size(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = micro_price_momentum(prices, window=3)
        # First 3 should be NaN
        assert np.all(np.isnan(result[:3]))
        assert not np.isnan(result[3])


class TestRollingHurstExponent:
    def test_random_walk(self):
        """Random walk should have H ≈ 0.5."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500))
        result = rolling_hurst_exponent(random_walk, window=100, max_lag=20)
        # Get the last valid Hurst value
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # Random walk should be approximately 0.5 (with some tolerance)
        assert 0.3 < np.mean(valid) < 0.7

    def test_output_length(self):
        data = np.random.randn(200)
        result = rolling_hurst_exponent(data, window=100, max_lag=20)
        assert len(result) == 200

    def test_short_input(self):
        data = np.random.randn(50)
        result = rolling_hurst_exponent(data, window=100)
        # All should be NaN since input < window
        assert np.all(np.isnan(result))


class TestRollingVolatility:
    def test_constant_returns(self):
        returns = np.ones(100) * 0.01
        result = rolling_volatility(returns, window=10)
        # Constant returns should have 0 volatility
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, 0.0)

    def test_increasing_volatility(self):
        low_vol = np.random.randn(50) * 0.01
        high_vol = np.random.randn(50) * 0.1
        returns = np.concatenate([low_vol, high_vol])
        result = rolling_volatility(returns, window=20)
        # Vol at end should be higher than vol near start
        assert result[-1] > result[25]


class TestVWAPDeviation:
    def test_price_at_vwap(self):
        prices = np.ones(100) * 100.0
        volumes = np.ones(100) * 10.0
        result = vwap_deviation(prices, volumes, window=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, 0.0)

    def test_price_above_vwap(self):
        prices = np.linspace(100, 110, 100)
        volumes = np.ones(100) * 10.0
        result = vwap_deviation(prices, volumes, window=10)
        # Latest price above VWAP → positive deviation
        assert result[-1] > 0


class TestTradeFlowImbalance:
    def test_all_buys(self):
        is_buyer_maker = np.array([False] * 100)  # All buy aggressor
        quantities = np.ones(100) * 1.0
        result = trade_flow_imbalance(is_buyer_maker, quantities, window=20)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, 1.0)

    def test_all_sells(self):
        is_buyer_maker = np.array([True] * 100)  # All sell aggressor
        quantities = np.ones(100) * 1.0
        result = trade_flow_imbalance(is_buyer_maker, quantities, window=20)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_almost_equal(valid, -1.0)


class TestFractionalDifferentiation:
    def test_output_length(self):
        series = np.random.randn(100).cumsum() + 100
        result = fractional_differentiation(series, d=0.5)
        assert len(result) == 100

    def test_d_zero_is_identity(self):
        """With d=0, fracdiff should return the original series (approximately)."""
        series = np.random.randn(50).cumsum() + 100
        result = fractional_differentiation(series, d=0.0)
        valid_mask = ~np.isnan(result)
        if valid_mask.any():
            np.testing.assert_array_almost_equal(
                result[valid_mask], series[valid_mask], decimal=3
            )

    def test_reduces_nonstationarity(self):
        """Fractional differentiation should reduce the variance of a trending series."""
        trend = np.arange(200, dtype=float)
        result = fractional_differentiation(trend, d=0.5)
        valid = result[~np.isnan(result)]
        # The variance should be less than the original trend
        assert np.var(valid) < np.var(trend)
