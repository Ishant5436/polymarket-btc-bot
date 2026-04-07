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
        self._lock = asyncio.Lock()
        self._last_price: float = 0.0
        self._trade_count: int = 0

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

    async def push_trade(self, trade_data: dict):
        """
        Push a new trade into the rolling buffer.
        
        Args:
            trade_data: Parsed trade dict from BinanceWebSocket with keys:
                price, quantity, timestamp, is_buyer_maker, trade_id
        """
        record = TradeRecord(
            price=trade_data["price"],
            quantity=trade_data["quantity"],
            timestamp=trade_data["timestamp"],
            is_buyer_maker=trade_data["is_buyer_maker"],
            trade_id=trade_data.get("trade_id", 0),
        )

        async with self._lock:
            self._trades.append(record)
            self._last_price = record.price
            self._trade_count += 1

    def push_trade_sync(self, trade_data: dict):
        """Synchronous version for testing / non-async contexts."""
        record = TradeRecord(
            price=trade_data["price"],
            quantity=trade_data["quantity"],
            timestamp=trade_data["timestamp"],
            is_buyer_maker=trade_data["is_buyer_maker"],
            trade_id=trade_data.get("trade_id", 0),
        )
        self._trades.append(record)
        self._last_price = record.price
        self._trade_count += 1

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

    def get_volatility(self, window_seconds: int) -> float:
        """
        Compute realized volatility (std of log returns) over a time window.
        """
        trades = self.get_window_by_time(window_seconds)
        if len(trades) < 3:
            return 0.0

        prices = np.array([t.price for t in trades], dtype=np.float64)
        log_rets = np.log(prices[1:] / prices[:-1])
        return float(np.std(log_rets)) if len(log_rets) > 0 else 0.0

    def clear(self):
        """Clear all trade data."""
        self._trades.clear()
        self._last_price = 0.0
        self._trade_count = 0
