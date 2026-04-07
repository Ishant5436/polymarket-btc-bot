"""Tests for the pre-live shadow qualification gate."""

from datetime import datetime, timezone

from src.exchange.gamma_api import MarketInfo
from src.execution.live_test_gate import LiveTestGate
from src.execution.order_router import TradingSignal
from src.utils.state import RollingState


def _iso_from_ms(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _push_trade(state: RollingState, price: float, timestamp_ms: int, trade_id: int):
    state.push_trade_sync(
        {
            "price": price,
            "quantity": 1.0,
            "timestamp": timestamp_ms,
            "is_buyer_maker": False,
            "trade_id": trade_id,
        }
    )


def test_live_test_gate_approves_after_profitable_shadow_window():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=600,
        min_completed_markets=2,
        min_win_rate=0.5,
        min_profit=0.1,
        max_cumulative_loss=0.0,
        now_fn=lambda: current_time["value"],
    )
    state = RollingState(maxlen=20)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 105.0, 300_000, 2)
    _push_trade(state, 100.0, 600_000, 3)

    market_1 = MarketInfo(
        condition_id="m1",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5m-1",
        yes_token_id="yes-1",
        no_token_id="no-1",
        end_date=_iso_from_ms(300_000),
    )
    market_2 = MarketInfo(
        condition_id="m2",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5m-2",
        yes_token_id="yes-2",
        no_token_id="no-2",
        end_date=_iso_from_ms(600_000),
    )

    gate.record_shadow_signal(
        market_1,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-1",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )
    gate.record_shadow_signal(
        market_2,
        TradingSignal(
            side="BUY_NO",
            token_id="no-2",
            price=0.45,
            size=1.0,
            edge=0.05,
            model_prob=0.45,
            market_price=0.50,
            timestamp=2.0,
        ),
    )

    current_time["value"] = 601.0
    settled = gate.settle_due_trades(state)
    status = gate.get_status()

    assert len(settled) == 2
    assert gate.allows_live_trading is True
    assert status["status"] == "approved"
    assert status["settled_markets"] == 2
    assert status["total_pnl"] > 0


def test_live_test_gate_fails_after_losing_shadow_window():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=300,
        min_completed_markets=1,
        min_win_rate=0.5,
        min_profit=0.01,
        max_cumulative_loss=0.0,
        now_fn=lambda: current_time["value"],
    )
    state = RollingState(maxlen=10)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 90.0, 300_000, 2)

    market = MarketInfo(
        condition_id="m1",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5m-loss",
        yes_token_id="yes-1",
        no_token_id="no-1",
        end_date=_iso_from_ms(300_000),
    )

    gate.record_shadow_signal(
        market,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-1",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )

    current_time["value"] = 301.0
    gate.settle_due_trades(state)
    status = gate.get_status()

    assert gate.allows_live_trading is False
    assert status["status"] == "failed"
    assert status["total_pnl"] < 0


def test_live_test_gate_settles_threshold_market_against_reference_price():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=300,
        min_completed_markets=1,
        min_win_rate=0.0,
        min_profit=-10.0,
        max_cumulative_loss=10.0,
        now_fn=lambda: current_time["value"],
    )
    state = RollingState(maxlen=10)
    _push_trade(state, 106.0, 300_000, 1)

    market = MarketInfo(
        condition_id="m-threshold",
        question="Bitcoin above 105 on April 6, 11AM ET?",
        slug="bitcoin-above-105-on-april-6-2026-11am-et",
        yes_token_id="yes-threshold",
        no_token_id="no-threshold",
        end_date=_iso_from_ms(300_000),
    )

    gate.record_shadow_signal(
        market,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-threshold",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )

    current_time["value"] = 301.0
    settled = gate.settle_due_trades(state)

    assert len(settled) == 1
    assert settled[0].won is True
    assert settled[0].resolution_type == "above"
    assert settled[0].reference_price == 105.0
    assert settled[0].start_price is None
    assert settled[0].end_price == 106.0


def test_live_test_gate_uses_market_interval_for_move_market_start():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=7200,
        min_completed_markets=1,
        min_win_rate=0.0,
        min_profit=-10.0,
        max_cumulative_loss=10.0,
        target_market_interval_minutes=60,
        now_fn=lambda: current_time["value"],
    )
    state = RollingState(maxlen=10)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 120.0, 3_300_000, 2)
    _push_trade(state, 110.0, 3_600_000, 3)

    market = MarketInfo(
        condition_id="m-hourly",
        question="Will BTC go up in 60 minutes?",
        slug="btc-60m-1",
        yes_token_id="yes-1",
        no_token_id="no-1",
        end_date=_iso_from_ms(3_600_000),
        market_interval_minutes=60,
    )

    gate.record_shadow_signal(
        market,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-1",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )

    current_time["value"] = 3_601.0
    settled = gate.settle_due_trades(state)

    assert len(settled) == 1
    assert settled[0].resolution_type == "move"
    assert settled[0].start_price == 100.0
    assert settled[0].end_price == 110.0
    assert settled[0].won is True


def test_live_test_gate_waits_for_pending_markets_after_window_closes():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=300,
        min_completed_markets=1,
        min_win_rate=0.0,
        min_profit=-10.0,
        max_cumulative_loss=10.0,
        now_fn=lambda: current_time["value"],
    )
    state = RollingState(maxlen=10)
    market = MarketInfo(
        condition_id="m-pending",
        question="Bitcoin above 105 on April 6, 11AM ET?",
        slug="bitcoin-above-105-on-april-6-2026-11am-et",
        yes_token_id="yes-threshold",
        no_token_id="no-threshold",
        end_date=_iso_from_ms(900_000),
    )

    gate.record_shadow_signal(
        market,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-threshold",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )

    current_time["value"] = 301.0
    settled = gate.settle_due_trades(state)
    status = gate.get_status()

    assert settled == []
    assert status["status"] == "pending"
    assert status["accepting_new_signals"] is False
    assert status["pending_markets"] == 1
    assert "waiting for 1 recorded market(s) to settle" in status["reason"]

    _push_trade(state, 106.0, 900_000, 1)
    current_time["value"] = 901.0
    settled = gate.settle_due_trades(state)
    status = gate.get_status()

    assert len(settled) == 1
    assert status["status"] == "approved"
    assert gate.allows_live_trading is True


def test_live_test_gate_rejects_new_signals_after_window_closes():
    current_time = {"value": 0.0}
    gate = LiveTestGate(
        qualification_window_seconds=300,
        min_completed_markets=1,
        min_win_rate=0.0,
        min_profit=-10.0,
        max_cumulative_loss=10.0,
        now_fn=lambda: current_time["value"],
    )
    market = MarketInfo(
        condition_id="m-late",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5m-late",
        yes_token_id="yes-late",
        no_token_id="no-late",
        end_date=_iso_from_ms(600_000),
    )

    current_time["value"] = 301.0
    recorded = gate.record_shadow_signal(
        market,
        TradingSignal(
            side="BUY_YES",
            token_id="yes-late",
            price=0.40,
            size=1.0,
            edge=0.05,
            model_prob=0.55,
            market_price=0.50,
            timestamp=1.0,
        ),
    )

    assert recorded is False
