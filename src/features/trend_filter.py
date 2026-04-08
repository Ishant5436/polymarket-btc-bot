"""
EMA trend confirmation filter.

Improvement 2: Blocks entries that disagree with the short-term trend
determined by EMA crossover. This cuts counter-trend losses by
preventing the bot from fighting momentum.

Uses 60-second sampled prices (minute-bar closes) instead of raw ticks
to eliminate bid-ask bounce noise from the trend signal.
"""

import logging
from typing import Optional

import numpy as np

from src.utils.state import RollingState

logger = logging.getLogger(__name__)

# Resample raw ticks into 60-second close prices for trend detection
_SAMPLE_INTERVAL_MS = 60_000


class TrendFilter:
    """
    Fast EMA / Slow EMA crossover gate.

    Only allows entries when the trade direction agrees with the
    short-term trend. This prevents buying YES during downtrends
    and buying NO during uptrends.

    Operates on 60-second sampled close prices to avoid noise from
    sub-second bid-ask bounce in raw tick data.
    """

    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 21,
        min_prices: int = 25,
    ):
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._min_prices = max(min_prices, slow_period + 1)

    def confirms_direction(self, side: str, state: RollingState) -> bool:
        """
        Return True when the EMA crossover agrees with the trade side.

        - BUY_YES: requires fast EMA > slow EMA (uptrend)
        - BUY_NO: requires fast EMA < slow EMA (downtrend)

        Returns True (permissive) when insufficient data is available.
        """
        # Fetch enough raw trades to cover slow_period minutes + buffer
        lookback_seconds = (self._min_prices + 5) * 60
        trades = state.get_window_by_time(lookback_seconds)
        if len(trades) < 3:
            return True  # Not enough data — be permissive

        # Down-sample raw ticks to 60-second interval closes
        prices = RollingState._sample_prices_by_interval(
            trades, _SAMPLE_INTERVAL_MS
        )
        if len(prices) < self._min_prices:
            return True

        # Only compute EMA on the tail we need (avoid O(100K) iteration)
        tail_len = self._slow_period * 3
        if len(prices) > tail_len:
            prices = prices[-tail_len:]

        fast_ema = self._ema(prices, self._fast_period)
        slow_ema = self._ema(prices, self._slow_period)

        if fast_ema is None or slow_ema is None:
            return True

        is_uptrend = fast_ema > slow_ema

        if side == "BUY_YES":
            if not is_uptrend:
                logger.debug(
                    "Trend filter blocked BUY_YES | fast_ema=%.2f slow_ema=%.2f",
                    fast_ema,
                    slow_ema,
                )
                return False
            return True

        if side == "BUY_NO":
            if is_uptrend:
                logger.debug(
                    "Trend filter blocked BUY_NO | fast_ema=%.2f slow_ema=%.2f",
                    fast_ema,
                    slow_ema,
                )
                return False
            return True

        return True

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> Optional[float]:
        """Compute the EMA of the last `period` prices."""
        if len(prices) < period:
            return None

        alpha = 2.0 / (period + 1)
        ema = float(prices[0])
        for price in prices[1:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        return ema
