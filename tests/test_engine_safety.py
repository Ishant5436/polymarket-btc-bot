"""Safety-focused tests for live engine initialization."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.exchange.gamma_api import MarketInfo
from src.execution.order_router import TradingSignal
from src.execution.engine import TradingEngine


def test_initialize_clients_blocks_live_without_explicit_enable(monkeypatch):
    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    engine = TradingEngine()

    assert engine._initialize_clients() is False


def test_initialize_clients_blocks_live_without_trading_access(monkeypatch):
    mock_client = MagicMock()
    mock_client.has_trading_access = False

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    engine = TradingEngine()

    assert engine._initialize_clients() is False


def test_initialize_clients_allows_dry_run_without_private_key(monkeypatch):
    mock_client = MagicMock()

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=True,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key=""),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    monkeypatch.setattr(
        "src.execution.engine.OrderRouter",
        lambda client, dry_run: SimpleNamespace(client=client, dry_run=dry_run),
    )
    monkeypatch.setattr(
        "src.execution.engine.RiskManager",
        lambda state, client, **kwargs: SimpleNamespace(
            state=state,
            client=client,
            **kwargs,
        ),
    )
    engine = TradingEngine()

    assert engine._initialize_clients() is True


def test_initialize_clients_honors_dry_run_override(monkeypatch):
    mock_client = MagicMock()

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key=""),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    monkeypatch.setattr(
        "src.execution.engine.OrderRouter",
        lambda client, dry_run: SimpleNamespace(client=client, dry_run=dry_run),
    )
    monkeypatch.setattr(
        "src.execution.engine.RiskManager",
        lambda state, client, **kwargs: SimpleNamespace(
            state=state,
            client=client,
            **kwargs,
        ),
    )
    engine = TradingEngine(dry_run_override=True)

    assert engine._initialize_clients() is True
    assert engine._router.dry_run is True
    assert engine._live_test_gate is None


def test_initialize_clients_creates_live_test_gate_for_live_mode(monkeypatch):
    mock_client = MagicMock()
    mock_client.has_trading_access = True
    mock_client.get_collateral_balance_allowance.return_value = SimpleNamespace(
        available_to_trade=10.0
    )
    mock_client.get_open_orders.return_value = []

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            require_live_test_before_live_orders=True,
            live_test_window_seconds=600,
            live_test_min_completed_markets=2,
            live_test_min_win_rate=0.5,
            live_test_min_profit=0.01,
            live_test_max_cumulative_loss=0.0,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    monkeypatch.setattr(
        "src.execution.engine.OrderRouter",
        lambda client, dry_run: SimpleNamespace(client=client, dry_run=dry_run),
    )
    monkeypatch.setattr(
        "src.execution.engine.RiskManager",
        lambda state, client, **kwargs: SimpleNamespace(
            state=state,
            client=client,
            **kwargs,
        ),
    )

    created = {}

    class FakeGate:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.qualification_window_seconds = kwargs["qualification_window_seconds"]
            self.min_completed_markets = kwargs["min_completed_markets"]
            self.min_win_rate = kwargs["min_win_rate"]
            self.min_profit = kwargs["min_profit"]
            self.max_cumulative_loss = kwargs["max_cumulative_loss"]

    monkeypatch.setattr("src.execution.engine.LiveTestGate", FakeGate)
    engine = TradingEngine()

    assert engine._initialize_clients() is True
    assert isinstance(engine._live_test_gate, FakeGate)
    assert created["qualification_window_seconds"] == 600
    assert created["min_completed_markets"] == 2


def test_initialize_clients_blocks_validation_only_without_trading_access(monkeypatch):
    mock_client = MagicMock()
    mock_client.has_trading_access = False

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=True,
            live_trading_enabled=False,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    engine = TradingEngine()

    assert engine._initialize_clients() is False


def test_initialize_clients_validation_only_forces_read_only_router(monkeypatch):
    mock_client = MagicMock()
    mock_client.has_trading_access = True
    mock_client.get_collateral_balance_allowance.return_value = SimpleNamespace(
        available_to_trade=10.0
    )
    mock_client.get_open_orders.return_value = []

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=True,
            live_trading_enabled=False,
            require_live_test_before_live_orders=True,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    monkeypatch.setattr(
        "src.execution.engine.OrderRouter",
        lambda client, dry_run: SimpleNamespace(client=client, dry_run=dry_run),
    )
    monkeypatch.setattr(
        "src.execution.engine.RiskManager",
        lambda state, client, **kwargs: SimpleNamespace(
            state=state,
            client=client,
            **kwargs,
        ),
    )
    engine = TradingEngine()

    assert engine._initialize_clients() is True
    assert engine._router.dry_run is True
    assert engine._live_test_gate is None


def test_validation_only_override_forces_read_only_mode(monkeypatch):
    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )

    engine = TradingEngine(validation_only_override=True)

    assert engine._is_validation_only_mode() is True
    assert engine._is_read_only_mode() is True


def test_market_supports_live_strategy_logs_hourly_override_only_once(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            allow_non_5m_live_markets=True,
        ),
    )
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="hourly-condition",
        question="Bitcoin above 69,600 on April 6, 2PM ET",
        slug="btc-hourly-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
        market_interval_minutes=60,
    )

    with caplog.at_level(logging.WARNING):
        assert engine._market_supports_live_strategy(market) is True
        assert engine._market_supports_live_strategy(market) is True

    messages = [
        record.message for record in caplog.records
        if "incompatible market horizon" in record.message
    ]
    assert len(messages) == 1


def test_market_supports_live_strategy_accepts_matching_hourly_model(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            allow_non_5m_live_markets=False,
        ),
    )
    engine = TradingEngine()
    engine._model = SimpleNamespace(target_horizon_minutes=60)
    market = MarketInfo(
        condition_id="hourly-condition",
        question="Will BTC go up in 60 minutes?",
        slug="btc-hourly-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T13:00:00Z",
        market_interval_minutes=60,
    )

    assert engine._market_supports_live_strategy(market) is True


def test_initialize_clients_blocks_live_when_no_spendable_collateral(monkeypatch):
    mock_client = MagicMock()
    mock_client.has_trading_access = True
    mock_client.get_collateral_balance_allowance.return_value = SimpleNamespace(
        available_to_trade=0.0
    )
    mock_client.get_open_orders.return_value = []

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            require_live_test_before_live_orders=True,
        ),
    )
    monkeypatch.setattr(
        "src.execution.engine.POLYMARKET",
        SimpleNamespace(private_key="test-private-key"),
    )
    monkeypatch.setattr("src.execution.engine.PolymarketClient", lambda: mock_client)
    engine = TradingEngine()

    assert engine._initialize_clients() is False


@pytest.mark.asyncio
async def test_run_inference_cycle_blocks_live_order_until_live_test_passes():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._risk = SimpleNamespace(run_all_checks=lambda **kwargs: True)
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._live_test_gate = MagicMock()
    engine._live_test_gate.allows_live_trading = False

    await engine._run_inference_cycle()

    engine._live_test_gate.settle_due_trades.assert_called_once_with(engine._state)
    engine._live_test_gate.record_shadow_signal.assert_called_once_with(market, signal)
    engine._router.execute_signal.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_progresses_shadow_gate_before_risk_checks():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    risk = MagicMock()
    risk.run_all_checks.return_value = False

    engine._risk = risk
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._live_test_gate = MagicMock()
    engine._live_test_gate.allows_live_trading = False

    await engine._run_inference_cycle()

    engine._live_test_gate.settle_due_trades.assert_called_once_with(engine._state)
    engine._live_test_gate.record_shadow_signal.assert_called_once_with(market, signal)
    risk.run_all_checks.assert_not_called()
    engine._router.execute_signal.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_executes_after_live_test_passes():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._risk = SimpleNamespace(run_all_checks=lambda **kwargs: True)
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": False},
    )
    engine._live_test_gate = MagicMock()
    engine._live_test_gate.allows_live_trading = True

    await engine._run_inference_cycle()

    engine._live_test_gate.settle_due_trades.assert_called_once_with(engine._state)
    engine._live_test_gate.record_shadow_signal.assert_not_called()
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_run_inference_cycle_skips_signal_polling_when_live_gate_window_is_closed():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )

    gate = MagicMock()
    gate.allows_live_trading = False
    gate.accepts_new_signals = False

    engine._active_market = market
    engine._risk = MagicMock()
    engine._pipeline = MagicMock()
    engine._model = MagicMock()
    engine._router = MagicMock()
    engine._live_test_gate = gate

    await engine._run_inference_cycle()

    gate.settle_due_trades.assert_called_once_with(engine._state)
    engine._pipeline.compute.assert_not_called()
    engine._router.get_signal.assert_not_called()
    engine._risk.run_all_checks.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_in_dry_run_skips_balance_precheck(monkeypatch):
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )
    risk = MagicMock()
    risk.run_all_checks.return_value = True

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=True,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )

    engine = TradingEngine()

    engine._risk = risk
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": True},
    )

    await engine._run_inference_cycle()

    risk.run_all_checks.assert_called_once_with(
        include_balance_check=False,
        include_position_limit=False,
    )


@pytest.mark.asyncio
async def test_run_inference_cycle_skips_signal_polling_when_risk_blocks(monkeypatch):
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    risk = MagicMock()
    risk.run_all_checks.return_value = False

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=True,
            validation_only_mode=False,
            live_trading_enabled=False,
        ),
    )

    engine = TradingEngine()
    engine._risk = risk
    engine._active_market = market
    engine._pipeline = MagicMock()
    engine._model = MagicMock()
    engine._router = MagicMock()

    await engine._run_inference_cycle()

    risk.run_all_checks.assert_called_once_with(
        include_balance_check=False,
        include_position_limit=False,
    )
    engine._pipeline.compute.assert_not_called()
    engine._router.get_signal.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_blocks_live_orders_on_hourly_market(monkeypatch):
    market = MarketInfo(
        condition_id="test-condition",
        question="Bitcoin above 70,200 on April 6, 1PM ET",
        slug="btc-hourly-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T13:00:00Z",
        market_interval_minutes=60,
    )

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
        ),
    )

    engine = TradingEngine()
    engine._active_market = market
    engine._risk = MagicMock()
    engine._pipeline = MagicMock()
    engine._model = MagicMock()
    engine._router = MagicMock()

    await engine._run_inference_cycle()

    engine._risk.run_all_checks.assert_not_called()
    engine._pipeline.compute.assert_not_called()
    engine._router.get_signal.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_allows_hourly_market_with_explicit_override(monkeypatch):
    market = MarketInfo(
        condition_id="test-condition",
        question="Bitcoin above 70,200 on April 6, 1PM ET",
        slug="btc-hourly-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T13:00:00Z",
        market_interval_minutes=60,
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            allow_non_5m_live_markets=True,
        ),
    )

    engine = TradingEngine()
    engine._active_market = market
    engine._risk = MagicMock()
    engine._risk.run_all_checks.return_value = True
    engine._pipeline = MagicMock()
    engine._pipeline.compute.return_value = np.zeros(20)
    engine._model = SimpleNamespace(
        target_horizon_minutes=5,
        predict=lambda features: 0.55,
    )
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": False},
    )

    await engine._run_inference_cycle()

    engine._risk.run_all_checks.assert_called_once()
    engine._pipeline.compute.assert_called_once_with()
    engine._router.get_signal.assert_called_once_with(0.55, market)
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_run_inference_cycle_allows_matching_hourly_model_without_override(
    monkeypatch,
):
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 60 minutes?",
        slug="btc-hourly-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T13:00:00Z",
        market_interval_minutes=60,
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=False,
            live_trading_enabled=True,
            allow_non_5m_live_markets=False,
        ),
    )

    engine = TradingEngine()
    engine._active_market = market
    engine._risk = MagicMock()
    engine._risk.run_all_checks.return_value = True
    engine._pipeline = MagicMock()
    engine._pipeline.compute.return_value = np.zeros(20)
    engine._model = SimpleNamespace(
        target_horizon_minutes=60,
        predict=lambda features: 0.55,
    )
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": False},
    )

    await engine._run_inference_cycle()

    engine._risk.run_all_checks.assert_called_once()
    engine._pipeline.compute.assert_called_once_with()
    engine._router.get_signal.assert_called_once_with(0.55, market)
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_run_inference_cycle_invalidates_private_risk_cache_after_order_attempt():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._active_market = market
    engine._risk = MagicMock()
    engine._risk.run_all_checks.return_value = True
    engine._pipeline = MagicMock()
    engine._pipeline.compute.return_value = np.zeros(20)
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": False},
    )

    await engine._run_inference_cycle()

    engine._risk.invalidate_private_check_cache.assert_called_once_with()
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_run_inference_cycle_keeps_private_risk_cache_for_duplicate_suppression():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._active_market = market
    engine._risk = MagicMock()
    engine._risk.run_all_checks.return_value = True
    engine._pipeline = MagicMock()
    engine._pipeline.compute.return_value = np.zeros(20)
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = None

    await engine._run_inference_cycle()

    engine._risk.invalidate_private_check_cache.assert_not_called()
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_run_inference_cycle_keeps_private_risk_cache_for_local_sizing_block():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._active_market = market
    engine._risk = MagicMock()
    engine._risk.run_all_checks.return_value = True
    engine._pipeline = MagicMock()
    engine._pipeline.compute.return_value = np.zeros(20)
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = SimpleNamespace(
        success=False,
        raw_response={"reason": "order_sizing_blocked"},
    )

    await engine._run_inference_cycle()

    engine._risk.invalidate_private_check_cache.assert_not_called()
    engine._router.execute_signal.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_risk_monitor_skips_volatility_checks_while_live_gate_is_locked(monkeypatch):
    engine = TradingEngine()
    engine._running = True
    engine._risk = MagicMock()
    engine._live_test_gate = SimpleNamespace(allows_live_trading=False)

    async def stop_after_one_sleep(_seconds):
        engine._running = False

    monkeypatch.setattr("src.execution.engine.asyncio.sleep", stop_after_one_sleep)

    await engine._risk_monitor_task()

    engine._risk.check_volatility.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_refreshes_expired_market_before_trading():
    engine = TradingEngine()
    expired_market = MarketInfo(
        condition_id="expired-condition",
        question="Expired market",
        slug="expired-market",
        yes_token_id="yes-token-old",
        no_token_id="no-token-old",
        end_date="2020-01-01T00:00:00Z",
    )
    refreshed_market = MarketInfo(
        condition_id="fresh-condition",
        question="Fresh market",
        slug="fresh-market",
        yes_token_id="yes-token-new",
        no_token_id="no-token-new",
        end_date="2099-01-01T00:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token-new",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._active_market = expired_market
    engine._gamma = MagicMock()
    engine._gamma.get_active_btc_5m_market.return_value = refreshed_market
    engine._risk = SimpleNamespace(run_all_checks=lambda **kwargs: True)
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = MagicMock(
        success=True,
        raw_response={"dry_run": False},
    )

    await engine._run_inference_cycle()

    engine._gamma.get_active_btc_5m_market.assert_called_once_with(force_refresh=True)
    assert engine._active_market == refreshed_market
    engine._router.get_signal.assert_called_once_with(0.55, refreshed_market)
    engine._router.execute_signal.assert_called_once_with(signal, refreshed_market)


@pytest.mark.asyncio
async def test_run_inference_cycle_does_not_record_position_for_live_blocked_dry_run():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._risk = SimpleNamespace(run_all_checks=lambda **kwargs: True)
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = SimpleNamespace(
        success=True,
        raw_response={"dry_run": True, "live_blocked": True},
    )
    engine._position_manager = MagicMock()

    await engine._run_inference_cycle()

    engine._position_manager.record_entry.assert_not_called()


@pytest.mark.asyncio
async def test_run_inference_cycle_records_position_for_fillable_dry_run():
    engine = TradingEngine()
    market = MarketInfo(
        condition_id="test-condition",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5min-test",
        yes_token_id="yes-token",
        no_token_id="no-token",
        end_date="2099-04-05T12:05:00Z",
    )
    signal = TradingSignal(
        side="BUY_YES",
        token_id="yes-token",
        price=0.49,
        size=1.0,
        edge=0.05,
        model_prob=0.55,
        market_price=0.50,
        timestamp=1_000.0,
    )

    engine._risk = SimpleNamespace(run_all_checks=lambda **kwargs: True)
    engine._active_market = market
    engine._pipeline = SimpleNamespace(compute=lambda: np.zeros(20))
    engine._model = SimpleNamespace(predict=lambda features: 0.55)
    engine._router = MagicMock()
    engine._router.get_signal.return_value = signal
    engine._router.execute_signal.return_value = SimpleNamespace(
        success=True,
        raw_response={"dry_run": True, "live_blocked": False},
    )
    engine._position_manager = MagicMock()

    await engine._run_inference_cycle()

    engine._position_manager.record_entry.assert_called_once_with(signal, market)


@pytest.mark.asyncio
async def test_shutdown_skips_cancel_all_orders_in_validation_only_mode(monkeypatch):
    monkeypatch.setattr(
        "src.execution.engine.TRADING",
        SimpleNamespace(
            dry_run=False,
            validation_only_mode=True,
            live_trading_enabled=False,
        ),
    )

    engine = TradingEngine()
    engine._running = True
    engine._pm_client = MagicMock()
    engine._binance_ws = SimpleNamespace(stop=AsyncMock())
    engine._gamma = MagicMock()

    await engine.shutdown()

    engine._pm_client.cancel_all_orders.assert_not_called()
    engine._binance_ws.stop.assert_awaited_once()
    engine._gamma.close.assert_called_once()


def test_sync_realized_pnl_uses_startup_baseline_then_tracks_deltas():
    engine = TradingEngine()
    engine._risk = MagicMock()
    engine._pm_client = MagicMock()

    engine._pm_client.get_current_positions.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "asset-1",
            "realizedPnl": 0.0,
            "timestamp": 1,
        },
        {
            "proxyWallet": "0xabc",
            "asset": "asset-2",
            "realizedPnl": 0.12,
            "timestamp": 2,
        },
    ]
    engine._pm_client.get_closed_positions.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "asset-2",
            "realizedPnl": 0.12,
            "timestamp": 2,
        },
        {
            "proxyWallet": "0xabc",
            "asset": "asset-3",
            "realizedPnl": -0.05,
            "timestamp": 3,
        },
    ]

    initial_delta = engine._sync_realized_pnl()

    assert initial_delta == pytest.approx(0.0)
    assert engine._risk.update_pnl.call_count == 0

    engine._pm_client.get_current_positions.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "asset-1",
            "realizedPnl": 0.02,
            "timestamp": 4,
        },
        {
            "proxyWallet": "0xabc",
            "asset": "asset-2",
            "realizedPnl": 0.12,
            "timestamp": 2,
        },
    ]
    engine._pm_client.get_closed_positions.return_value = [
        {
            "proxyWallet": "0xabc",
            "asset": "asset-3",
            "realizedPnl": -0.08,
            "timestamp": 5,
        },
    ]

    second_delta = engine._sync_realized_pnl()

    assert second_delta == pytest.approx(-0.01)
    assert engine._risk.update_pnl.call_count == 1
    assert engine._risk.update_pnl.call_args_list[0].args[0] == pytest.approx(-0.01)
