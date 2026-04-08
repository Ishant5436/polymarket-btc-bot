"""Tests for feature engineering indicators."""

from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from src.features.indicators import (
    fractional_differentiation,
    micro_price_momentum,
    order_book_imbalance,
    rolling_hurst_exponent,
    rolling_volatility,
    trade_flow_imbalance,
    vwap_deviation,
)
from src.features.minute_features import aggregate_trades_to_1m_bars, compute_feature_frame


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


class TestPseudoLiquidations:
    def test_sell_side_flush_creates_positive_pseudo_liq_imbalance(self):
        bars = pd.DataFrame(
            [
                {
                    "open_time": pd.Timestamp("2024-01-01 00:00:00"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.5,
                    "close": 100.2,
                    "volume": 10.0,
                    "taker_buy_base": 5.0,
                    "trades_count": 20,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
                {
                    "open_time": pd.Timestamp("2024-01-01 00:01:00"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 93.0,
                    "close": 94.0,
                    "volume": 120.0,
                    "taker_buy_base": 2.0,
                    "trades_count": 160,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
            ]
        )

        features = compute_feature_frame(bars)

        assert features.iloc[-1]["pseudo_liq_imbalance_1m"] > 0
        assert (
            features.iloc[-1]["pseudo_liq_imbalance_1m"]
            > features.iloc[0]["pseudo_liq_imbalance_1m"]
        )

    def test_buy_side_squeeze_creates_negative_pseudo_liq_imbalance(self):
        bars = pd.DataFrame(
            [
                {
                    "open_time": pd.Timestamp("2024-01-01 00:00:00"),
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.8,
                    "close": 100.1,
                    "volume": 12.0,
                    "taker_buy_base": 6.0,
                    "trades_count": 18,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
                {
                    "open_time": pd.Timestamp("2024-01-01 00:01:00"),
                    "open": 100.0,
                    "high": 108.0,
                    "low": 99.4,
                    "close": 107.5,
                    "volume": 130.0,
                    "taker_buy_base": 127.0,
                    "trades_count": 170,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
            ]
        )

        features = compute_feature_frame(bars)

        assert features.iloc[-1]["pseudo_liq_imbalance_1m"] < 0

    def test_quiet_balanced_bar_keeps_pseudo_liq_near_zero(self):
        bars = pd.DataFrame(
            [
                {
                    "open_time": pd.Timestamp("2024-01-01 00:00:00"),
                    "open": 100.0,
                    "high": 100.3,
                    "low": 99.9,
                    "close": 100.1,
                    "volume": 20.0,
                    "taker_buy_base": 10.0,
                    "trades_count": 25,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
                {
                    "open_time": pd.Timestamp("2024-01-01 00:01:00"),
                    "open": 100.1,
                    "high": 100.4,
                    "low": 99.95,
                    "close": 100.0,
                    "volume": 21.0,
                    "taker_buy_base": 10.5,
                    "trades_count": 24,
                    "liq_long_notional": 0.0,
                    "liq_short_notional": 0.0,
                },
            ]
        )

        features = compute_feature_frame(bars)

        assert abs(features.iloc[-1]["pseudo_liq_imbalance_1m"]) < 0.05


class TestMinuteBarAggregation:
    def test_aggregate_trades_to_1m_bars_merges_liquidations_without_series_view(self):
        trades = [
            SimpleNamespace(
                timestamp=1_710_000_000_000,
                price=100.0,
                quantity=1.0,
                is_buyer_maker=False,
            ),
            SimpleNamespace(
                timestamp=1_710_000_030_000,
                price=101.0,
                quantity=2.0,
                is_buyer_maker=True,
            ),
            SimpleNamespace(
                timestamp=1_710_000_060_000,
                price=102.0,
                quantity=1.5,
                is_buyer_maker=False,
            ),
            SimpleNamespace(
                timestamp=1_710_000_119_500,
                price=103.0,
                quantity=1.0,
                is_buyer_maker=False,
            ),
        ]
        liquidations = [
            SimpleNamespace(
                timestamp=1_710_000_010_000,
                price=99.0,
                quantity=3.0,
                side="SELL",
            ),
            SimpleNamespace(
                timestamp=1_710_000_090_000,
                price=104.0,
                quantity=2.0,
                side="BUY",
            ),
        ]

        bars = aggregate_trades_to_1m_bars(
            trades,
            liquidations,
            include_incomplete_last_bar=True,
        )

        assert len(bars) == 2
        assert bars["liq_long_notional"].tolist() == pytest.approx([297.0, 0.0])
        assert bars["liq_short_notional"].tolist() == pytest.approx([0.0, 208.0])
