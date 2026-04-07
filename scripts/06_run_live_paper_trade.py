#!/usr/bin/env python3
"""
Run a true live paper trade and wait for settlement.

The script:
1. Connects to live Binance and Polymarket market data
2. Uses the trained model to generate the first live paper-trade signal
3. Records that signal without placing a real order
4. Waits for the market to resolve and prints the settled P&L
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exchange.binance_rest import BinanceRESTClient
from src.exchange.binance_ws import BinanceWebSocket
from src.exchange.gamma_api import GammaAPIClient, MarketInfo
from src.exchange.polymarket_client import PolymarketClient
from src.execution.inference import ModelInference
from src.execution.live_test_gate import LiveTestGate, SettledShadowTrade
from src.execution.order_router import OrderRouter, TradingSignal
from src.execution.probability_estimator import MarketProbabilityEstimator
from src.features.pipeline import FeaturePipeline
from src.utils.logging_config import setup_logging
from src.utils.state import RollingState


class LivePaperTradeRunner:
    """Capture and settle one live paper trade using read-only market data."""

    def __init__(self, budget: float, timeout_seconds: int):
        self._budget = budget
        self._timeout_seconds = timeout_seconds
        self._state = RollingState()
        self._pipeline = FeaturePipeline(self._state)
        self._model = ModelInference()
        self._probability_estimator = MarketProbabilityEstimator()
        self._binance_rest = BinanceRESTClient()
        self._binance_ws = BinanceWebSocket()
        self._gamma = GammaAPIClient()
        self._pm_client = PolymarketClient()
        self._router = OrderRouter(
            client=self._pm_client,
            dry_run=True,
            order_notional=budget,
            max_order_notional=budget,
            bankroll_fraction_per_order=1.0,
            allow_upsize_to_min_order_size=False,
        )
        self._gate = LiveTestGate(
            qualification_window_seconds=10**9,
            min_completed_markets=1,
            min_win_rate=0.0,
            min_profit=-10**9,
            max_cumulative_loss=10**9,
        )
        self._active_market: Optional[MarketInfo] = None
        self._recorded_signal: Optional[TradingSignal] = None
        self._running = False
        self._start_ts = time.time()

    def _seed_pipeline_history(self):
        """Backfill closed minute bars so the first inference can happen quickly."""
        target_bars = self._pipeline.required_complete_bars + 18
        bars = self._binance_rest.fetch_recent_1m_klines(limit=target_bars)
        if bars.empty:
            raise RuntimeError("Binance warmup returned no closed minute bars")

        self._pipeline.seed_historical_bars(bars)

        # Seed the rolling state with recent bar closes so settlement can resolve
        # even if the market expires shortly after startup.
        for index, row in bars.iterrows():
            close_timestamp_ms = int(row["open_time"].timestamp() * 1000) + 59_000
            self._state.push_trade_sync(
                {
                    "price": float(row["close"]),
                    "quantity": float(row["volume"]) if row["volume"] else 1.0,
                    "timestamp": close_timestamp_ms,
                    "is_buyer_maker": False,
                    "trade_id": int(index),
                }
            )

    async def _state_ingestion_task(self):
        """Move parsed Binance trades from the WS queue into rolling state."""
        queue = self._binance_ws.queue
        while self._running:
            try:
                trade = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            await self._state.push_trade(trade)

    async def _market_discovery_task(self):
        """Refresh the currently active BTC market periodically."""
        while self._running:
            try:
                market = self._gamma.get_active_btc_5m_market(force_refresh=True)
                if market is not None:
                    self._active_market = market
            except Exception:
                pass
            await asyncio.sleep(30)

    async def _strategy_task(self):
        """Record one signal and then wait for it to settle."""
        while self._running:
            if time.time() - self._start_ts > self._timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {self._timeout_seconds}s waiting for a settled paper trade"
                )

            settled = self._gate.settle_due_trades(self._state)
            if settled:
                self._running = False
                return

            if self._recorded_signal is not None:
                await asyncio.sleep(1)
                continue

            if self._active_market is None or not self._pipeline.is_ready:
                await asyncio.sleep(1)
                continue

            features = self._pipeline.compute()
            if features is None:
                await asyncio.sleep(1)
                continue

            raw_prob = self._model.predict(features)
            if raw_prob is None:
                await asyncio.sleep(1)
                continue

            prob = self._probability_estimator.estimate_yes_probability(
                raw_prob,
                self._active_market,
                self._state,
            )
            signal = self._router.get_signal(prob, self._active_market)
            if signal is None:
                await asyncio.sleep(1)
                continue

            recorded = self._gate.record_shadow_signal(self._active_market, signal)
            if recorded:
                self._recorded_signal = signal
                print(
                    f"Recorded paper trade | market={self._active_market.slug} "
                    f"question={self._active_market.question} side={signal.side} "
                    f"entry_price={signal.price:.2f} size={signal.size:.4f} "
                    f"budget=${self._budget:.2f}",
                    flush=True,
                )
            await asyncio.sleep(1)

    async def run(self) -> SettledShadowTrade:
        """Start the live paper-trade run and return the first settled trade."""
        if not self._model.load():
            raise RuntimeError("Model failed to load")

        self._seed_pipeline_history()
        self._running = True

        ws_task = asyncio.create_task(self._binance_ws.start(), name="binance_ws")
        ingest_task = asyncio.create_task(self._state_ingestion_task(), name="state_ingest")
        market_task = asyncio.create_task(self._market_discovery_task(), name="market_discovery")
        strategy_task = asyncio.create_task(self._strategy_task(), name="strategy")
        tasks = [ws_task, ingest_task, market_task, strategy_task]

        try:
            await strategy_task
            settled = self._gate.settled_trades
            if not settled:
                raise RuntimeError("No paper trade settled before the runner stopped")
            return settled[0]
        finally:
            self._running = False
            await self._binance_ws.stop()
            self._binance_rest.close()
            self._gamma.close()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def _format_settlement(trade: SettledShadowTrade, budget: float) -> str:
    profit_margin_pct = (trade.pnl / budget * 100.0) if budget > 0 else 0.0
    reference_text = (
        f"{trade.resolution_type} {trade.reference_price:.2f}"
        if trade.reference_price is not None
        else "5-minute move"
    )
    return (
        f"Settled paper trade | market={trade.market_slug} "
        f"question={trade.market_question} side={trade.side} won={trade.won} "
        f"entry_price={trade.entry_price:.2f} end_price={trade.end_price:.2f} "
        f"resolution={reference_text} pnl=${trade.pnl:.4f} "
        f"profit_margin_pct={profit_margin_pct:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run one real-time settled paper trade")
    parser.add_argument("--budget", type=float, default=3.0, help="Paper-trade dollar budget")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="How long to wait for the first settled trade",
    )
    args = parser.parse_args()

    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_output=False,
    )

    runner = LivePaperTradeRunner(
        budget=args.budget,
        timeout_seconds=args.timeout_seconds,
    )
    settled_trade = asyncio.run(runner.run())
    print(_format_settlement(settled_trade, args.budget))
    print(asdict(settled_trade))


if __name__ == "__main__":
    main()
