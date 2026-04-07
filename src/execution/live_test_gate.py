"""
Fail-closed live qualification gate.

Before any real orders are allowed, the bot can shadow the live BTC market,
settle those paper trades against observed BTC price movement, and only unlock
live execution if the configured qualification metrics are met.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from src.exchange.gamma_api import MarketInfo
from src.execution.market_rules import (
    MarketResolutionRule,
    derive_market_resolution_rule,
    settles_yes,
)
from src.execution.order_router import TradingSignal
from src.utils.model_metadata import DEFAULT_TARGET_HORIZON_MINUTES
from src.utils.state import RollingState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowTrade:
    """A shadow trade recorded during the live qualification period."""
    condition_id: str
    market_slug: str
    market_question: str
    side: str
    entry_price: float
    size: float
    model_prob: float
    market_start_ms: int
    market_end_ms: int
    resolution_type: str
    reference_price: Optional[float]
    recorded_at: float


@dataclass(frozen=True)
class SettledShadowTrade:
    """A shadow trade resolved against observed BTC market direction."""
    condition_id: str
    market_slug: str
    market_question: str
    side: str
    entry_price: float
    size: float
    start_price: Optional[float]
    end_price: float
    resolution_type: str
    reference_price: Optional[float]
    pnl: float
    won: bool
    resolved_at: float


class LiveTestGate:
    """
    Blocks live orders until a configurable shadow-trading window passes.

    Important: this improves discipline and reduces premature live execution,
    but it still cannot guarantee future profits or prevent all losses.
    """

    def __init__(
        self,
        qualification_window_seconds: int,
        min_completed_markets: int,
        min_win_rate: float,
        min_profit: float,
        max_cumulative_loss: float,
        target_market_interval_minutes: int = DEFAULT_TARGET_HORIZON_MINUTES,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self._qualification_window_seconds = qualification_window_seconds
        self._min_completed_markets = min_completed_markets
        self._min_win_rate = min_win_rate
        self._min_profit = min_profit
        self._max_cumulative_loss = max_cumulative_loss
        self._target_market_interval_minutes = max(1, int(target_market_interval_minutes))
        self._now_fn = now_fn or time.time
        self._started_at = self._now_fn()
        self._status = "pending"
        self._status_reason = "Live qualification window is still running"
        self._pending_trades: dict[str, ShadowTrade] = {}
        self._settled_trades: list[SettledShadowTrade] = []

    @property
    def qualification_window_seconds(self) -> int:
        return self._qualification_window_seconds

    @property
    def min_completed_markets(self) -> int:
        return self._min_completed_markets

    @property
    def min_win_rate(self) -> float:
        return self._min_win_rate

    @property
    def min_profit(self) -> float:
        return self._min_profit

    @property
    def max_cumulative_loss(self) -> float:
        return self._max_cumulative_loss

    @property
    def allows_live_trading(self) -> bool:
        return self._status == "approved"

    @property
    def accepts_new_signals(self) -> bool:
        """
        Return True while the qualification recording window is still open.

        After the window closes, the gate should keep waiting for already
        recorded shadow trades to settle, but it should stop admitting new ones.
        """
        return (
            self._status == "pending"
            and (self._now_fn() - self._started_at) < self._qualification_window_seconds
        )

    @property
    def total_pnl(self) -> float:
        return sum(trade.pnl for trade in self._settled_trades)

    @property
    def settled_trades(self) -> list[SettledShadowTrade]:
        return list(self._settled_trades)

    def record_shadow_signal(self, market: MarketInfo, signal: TradingSignal) -> bool:
        """
        Record one shadow trade per market for qualification scoring.
        """
        if not self.accepts_new_signals:
            return False

        if market.condition_id in self._pending_trades:
            return False

        if any(
            trade.condition_id == market.condition_id for trade in self._settled_trades
        ):
            return False

        market_end_ms = self._parse_market_end_ms(market.end_date)
        if market_end_ms is None:
            logger.warning(
                "Skipping live-test shadow trade: cannot parse market end date | slug=%s end=%s",
                market.slug,
                market.end_date,
            )
            return False

        market_window_minutes = self._resolve_market_window_minutes(market)
        market_start_ms = int(
            (
                datetime.fromtimestamp(market_end_ms / 1000, tz=timezone.utc)
                - timedelta(minutes=market_window_minutes)
            ).timestamp()
            * 1000
        )
        rule = derive_market_resolution_rule(market)

        self._pending_trades[market.condition_id] = ShadowTrade(
            condition_id=market.condition_id,
            market_slug=market.slug,
            market_question=market.question,
            side=signal.side,
            entry_price=signal.price,
            size=signal.size,
            model_prob=signal.model_prob,
            market_start_ms=market_start_ms,
            market_end_ms=market_end_ms,
            resolution_type=rule.resolution_type,
            reference_price=rule.reference_price,
            recorded_at=self._now_fn(),
        )
        logger.info(
            "Live-test shadow trade recorded | market=%s side=%s price=%.2f size=%.2f "
            "resolution=%s reference=%s",
            market.slug or market.condition_id,
            signal.side,
            signal.price,
            signal.size,
            rule.resolution_type,
            f"{rule.reference_price:.2f}" if rule.reference_price is not None else "N/A",
        )
        return True

    def settle_due_trades(self, state: RollingState) -> list[SettledShadowTrade]:
        """
        Settle any shadow trades whose market window has resolved.
        """
        if self._status == "approved":
            return []

        settled: list[SettledShadowTrade] = []

        for condition_id, trade in list(self._pending_trades.items()):
            if state.latest_timestamp_ms < trade.market_end_ms:
                continue

            end_price = state.get_price_at_or_before(trade.market_end_ms)
            if end_price is None:
                continue

            start_price: Optional[float]
            if trade.resolution_type == "move":
                start_price = state.get_price_at_or_before(trade.market_start_ms)
                if start_price is None:
                    continue
            else:
                start_price = None

            try:
                yes_won = settles_yes(
                    MarketResolutionRule(
                        resolution_type=trade.resolution_type,
                        reference_price=trade.reference_price,
                    ),
                    end_price=end_price,
                    start_price=start_price,
                )
            except ValueError:
                continue

            won = (
                (trade.side == "BUY_YES" and yes_won)
                or (trade.side == "BUY_NO" and not yes_won)
            )
            pnl = ((1.0 - trade.entry_price) if won else -trade.entry_price) * trade.size

            result = SettledShadowTrade(
                condition_id=trade.condition_id,
                market_slug=trade.market_slug,
                market_question=trade.market_question,
                side=trade.side,
                entry_price=trade.entry_price,
                size=trade.size,
                start_price=start_price,
                end_price=end_price,
                resolution_type=trade.resolution_type,
                reference_price=trade.reference_price,
                pnl=pnl,
                won=won,
                resolved_at=self._now_fn(),
            )
            self._settled_trades.append(result)
            settled.append(result)
            del self._pending_trades[condition_id]

            logger.info(
                "Live-test shadow trade settled | market=%s side=%s won=%s pnl=%.4f "
                "resolution=%s reference=%s start=%s end=%.2f",
                result.market_slug or result.condition_id,
                result.side,
                result.won,
                result.pnl,
                result.resolution_type,
                (
                    f"{result.reference_price:.2f}"
                    if result.reference_price is not None
                    else "N/A"
                ),
                (
                    f"{result.start_price:.2f}"
                    if result.start_price is not None
                    else "N/A"
                ),
                result.end_price,
            )

        self._evaluate_status()
        return settled

    def _resolve_market_window_minutes(self, market: MarketInfo) -> int:
        """Prefer the discovered market interval, falling back to the model target."""
        interval = getattr(market, "market_interval_minutes", None)
        if isinstance(interval, bool):
            parsed = 0
        elif isinstance(interval, (int, float, str)):
            try:
                parsed = int(interval)
            except (TypeError, ValueError):
                parsed = 0
        else:
            parsed = 0
        if parsed > 0:
            return parsed
        return self._target_market_interval_minutes

    def get_status(self) -> dict:
        """Return a compact status snapshot for logs and monitoring."""
        settled = len(self._settled_trades)
        wins = sum(1 for trade in self._settled_trades if trade.won)
        return {
            "status": self._status,
            "reason": self._status_reason,
            "elapsed_seconds": round(self._now_fn() - self._started_at, 1),
            "accepting_new_signals": self.accepts_new_signals,
            "pending_markets": len(self._pending_trades),
            "settled_markets": settled,
            "win_rate": round(wins / settled, 4) if settled else 0.0,
            "total_pnl": round(self.total_pnl, 4),
            "worst_cumulative_pnl": round(self._worst_cumulative_pnl(), 4),
            "allows_live_trading": self.allows_live_trading,
        }

    def _evaluate_status(self):
        if self._status != "pending":
            return

        elapsed = self._now_fn() - self._started_at
        if elapsed < self._qualification_window_seconds:
            return

        if self._pending_trades:
            self._status_reason = (
                "Qualification recording window closed; waiting for "
                f"{len(self._pending_trades)} recorded market(s) to settle"
            )
            return

        settled = len(self._settled_trades)
        if settled < self._min_completed_markets:
            self._status = "failed"
            self._status_reason = (
                f"Only {settled} live-test markets settled during the qualification window"
            )
            logger.error("Live-test gate failed | %s", self._status_reason)
            return

        wins = sum(1 for trade in self._settled_trades if trade.won)
        win_rate = wins / settled if settled else 0.0
        total_pnl = self.total_pnl
        worst_cumulative_pnl = self._worst_cumulative_pnl()

        if (
            win_rate >= self._min_win_rate
            and total_pnl >= self._min_profit
            and worst_cumulative_pnl >= -self._max_cumulative_loss
        ):
            self._status = "approved"
            self._status_reason = (
                f"Passed live qualification with win_rate={win_rate:.2%} "
                f"and pnl={total_pnl:.4f}"
            )
            logger.warning("Live-test gate approved | %s", self._status_reason)
            return

        self._status = "failed"
        self._status_reason = (
            "Failed live qualification: "
            f"win_rate={win_rate:.2%}, pnl={total_pnl:.4f}, "
            f"worst_cumulative_pnl={worst_cumulative_pnl:.4f}"
        )
        logger.error("Live-test gate failed | %s", self._status_reason)

    def _worst_cumulative_pnl(self) -> float:
        cumulative = 0.0
        worst = 0.0
        for trade in self._settled_trades:
            cumulative += trade.pnl
            worst = min(worst, cumulative)
        return worst

    @staticmethod
    def _parse_market_end_ms(end_date: str) -> Optional[int]:
        """Parse the market's ISO end timestamp into epoch milliseconds."""
        if not end_date:
            return None

        normalized = end_date.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
