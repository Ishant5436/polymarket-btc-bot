"""
Rolling state management for real-time trade data.
Uses collections.deque for O(1) append/pop with fixed max length.
Provides fast access to windowed data for feature computation.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.settings import FEATURES


@dataclass
class TradeRecord:
    """A single parsed trade record."""
    price: float
    quantity: float
    timestamp: int       # ms since epoch
    is_buyer_maker: bool # True = sell aggressor
    trade_id: int = 0


@dataclass
class LiquidationRecord:
    """A forced liquidation event from Binance Futures."""
    side: str            # "BUY" or "SELL"
    price: float
    quantity: float
    timestamp: int       # ms since epoch


class RollingState:
    """
    High-performance rolling window of recent trades.
    Used by the feature pipeline to compute real-time indicators
    matching the model's exact input shape.
    """

    def __init__(self, maxlen: Optional[int] = None):
        self._maxlen = (
            maxlen if maxlen is not None else FEATURES.rolling_state_maxlen
        )
        self._trades: deque[TradeRecord] = deque(maxlen=self._maxlen)
        self._liquidations: deque[LiquidationRecord] = deque(maxlen=self._maxlen)
        self._lock = asyncio.Lock()
        self._last_price: float = 0.0
        self._trade_count: int = 0
        self._liquidation_count: int = 0

    @property
    def size(self) -> int:
        """Current number of trades in the buffer."""
        return len(self._trades)

    @property
    def maxlen(self) -> int:
        """Configured maximum buffer size."""
        return self._maxlen

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data for feature computation."""
        return (
            len(self._trades) >= FEATURES.minimum_warmup_trades
            and self.history_span_seconds >= FEATURES.minimum_history_seconds
        )

    @property
    def last_price(self) -> float:
        """Most recently received price."""
        return self._last_price

    @property
    def latest_timestamp_ms(self) -> int:
        """Latest trade timestamp in milliseconds, or 0 if empty."""
        if not self._trades:
            return 0
        return self._trades[-1].timestamp

    @property
    def oldest_timestamp_ms(self) -> int:
        """Oldest trade timestamp in milliseconds, or 0 if empty."""
        if not self._trades:
            return 0
        return self._trades[0].timestamp

    @property
    def history_span_seconds(self) -> float:
        """Covered history span, measured from oldest to newest buffered trade."""
        if len(self._trades) < 2:
            return 0.0
        return (self.latest_timestamp_ms - self.oldest_timestamp_ms) / 1000.0

    @property
    def trade_count(self) -> int:
        """Total trades received since initialization."""
        return self._trade_count

    async def push_event(self, event_data: dict):
        """
        Push a new trade or liquidation into the rolling buffer.
        """
        async with self._lock:
            if event_data.get("type") == "liquidation":
                record = LiquidationRecord(
                    side=event_data["side"],
                    price=event_data["price"],
                    quantity=event_data["quantity"],
                    timestamp=event_data["timestamp"],
                )
                self._liquidations.append(record)
                self._liquidation_count += 1
            else:
                record = TradeRecord(
                    price=event_data["price"],
                    quantity=event_data["quantity"],
                    timestamp=event_data["timestamp"],
                    is_buyer_maker=event_data["is_buyer_maker"],
                    trade_id=event_data.get("trade_id", 0),
                )
                self._trades.append(record)
                self._last_price = record.price
                self._trade_count += 1

    def push_event_sync(self, event_data: dict):
        """Synchronous version for testing / non-async contexts."""
        if event_data.get("type") == "liquidation":
            record = LiquidationRecord(
                side=event_data["side"],
                price=event_data["price"],
                quantity=event_data["quantity"],
                timestamp=event_data["timestamp"],
            )
            self._liquidations.append(record)
            self._liquidation_count += 1
        else:
            record = TradeRecord(
                price=event_data["price"],
                quantity=event_data["quantity"],
                timestamp=event_data["timestamp"],
                is_buyer_maker=event_data["is_buyer_maker"],
                trade_id=event_data.get("trade_id", 0),
            )
            self._trades.append(record)
            self._last_price = record.price
            self._trade_count += 1

    def push_trade_sync(self, event_data: dict):
        """Compatibility wrapper for tests that push plain trade payloads."""
        self.push_event_sync(event_data)

    def push_liquidation_sync(self, event_data: dict):
        """Compatibility wrapper for tests that push liquidation payloads."""
        liquidation_event = dict(event_data)
        liquidation_event["type"] = "liquidation"
        self.push_event_sync(liquidation_event)

    def get_liquidations(self) -> list[LiquidationRecord]:
        """Get all buffered liquidations."""
        return list(self._liquidations)

    def get_prices(self, n: Optional[int] = None) -> np.ndarray:
        """Get array of recent prices (most recent last)."""
        trades = list(self._trades)
        if n is not None:
            trades = trades[-n:]
        return np.array([t.price for t in trades], dtype=np.float64)

    def get_trades(self, n: Optional[int] = None) -> list[TradeRecord]:
        """Get recent trades in chronological order."""
        trades = list(self._trades)
        if n is not None:
            trades = trades[-n:]
        return trades

    def get_quantities(self, n: Optional[int] = None) -> np.ndarray:
        """Get array of recent quantities."""
        trades = list(self._trades)
        if n is not None:
            trades = trades[-n:]
        return np.array([t.quantity for t in trades], dtype=np.float64)

    def get_timestamps(self, n: Optional[int] = None) -> np.ndarray:
        """Get array of recent timestamps (ms)."""
        trades = list(self._trades)
        if n is not None:
            trades = trades[-n:]
        return np.array([t.timestamp for t in trades], dtype=np.int64)

    def get_buyer_maker_flags(self, n: Optional[int] = None) -> np.ndarray:
        """Get array of is_buyer_maker flags."""
        trades = list(self._trades)
        if n is not None:
            trades = trades[-n:]
        return np.array([t.is_buyer_maker for t in trades], dtype=bool)

    def get_window_by_time(self, seconds: int) -> list[TradeRecord]:
        """
        Get trades from the last `seconds` seconds.
        Uses the latest buffered trade timestamp as the reference point.
        """
        if not self._trades:
            return []

        cutoff_ms = self.latest_timestamp_ms - (seconds * 1000)
        result = []
        for trade in reversed(self._trades):
            if trade.timestamp >= cutoff_ms:
                result.append(trade)
            else:
                break

        result.reverse()  # Maintain chronological order
        return result

    def get_log_returns(self, n: Optional[int] = None) -> np.ndarray:
        """Compute log returns from the price series."""
        prices = self.get_prices(n)
        if len(prices) < 2:
            return np.array([], dtype=np.float64)
        return np.log(prices[1:] / prices[:-1])

    def get_price_at_or_before(self, timestamp_ms: int) -> Optional[float]:
        """
        Return the most recent buffered trade price at or before `timestamp_ms`.
        """
        for trade in reversed(self._trades):
            if trade.timestamp <= timestamp_ms:
                return trade.price
        return None

    @staticmethod
    def _sample_prices_by_interval(
        trades: list[TradeRecord],
        sample_interval_ms: int,
    ) -> np.ndarray:
        """
        Collapse raw trades into interval-close prices.

        Using the last trade in each interval makes volatility estimates less
        sensitive to intra-second bid/ask bounce while still reacting to actual
        directional moves in the underlying tape.
        """
        if sample_interval_ms <= 0:
            return np.array([trade.price for trade in trades], dtype=np.float64)

        sampled_prices: list[float] = []
        current_bucket: Optional[int] = None

        for trade in trades:
            bucket = trade.timestamp // sample_interval_ms
            if bucket != current_bucket:
                sampled_prices.append(trade.price)
                current_bucket = bucket
            else:
                sampled_prices[-1] = trade.price

        return np.array(sampled_prices, dtype=np.float64)

    def get_volatility(self, window_seconds: int) -> float:
        """
        Compute realized volatility over a time window using 1-second closes.

        The risk layer cares about regime shifts in BTC, not raw tick-by-tick
        microstructure bounce. Sampling the last trade per second provides a
        more stable realized-vol estimate for live kill-switch decisions.
        """
        trades = self.get_window_by_time(window_seconds)
        if len(trades) < 3:
            return 0.0

        prices = self._sample_prices_by_interval(trades, sample_interval_ms=1000)
        if len(prices) < 3:
            return 0.0

        log_rets = np.log(prices[1:] / prices[:-1])
        return float(np.std(log_rets)) if len(log_rets) > 0 else 0.0

    def clear(self):
        """Clear all trade and liquidation data."""
        self._trades.clear()
        self._liquidations.clear()
        self._last_price = 0.0
        self._trade_count = 0
        self._liquidation_count = 0
