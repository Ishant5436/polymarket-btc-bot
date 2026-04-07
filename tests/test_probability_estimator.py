from datetime import datetime, timedelta, timezone

from src.exchange.gamma_api import MarketInfo
from src.execution.probability_estimator import MarketProbabilityEstimator
from src.utils.state import RollingState


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


def test_move_market_uses_raw_direction_probability():
    estimator = MarketProbabilityEstimator()
    state = RollingState(maxlen=10)
    _push_trade(state, 100.0, 0, 1)
    _push_trade(state, 100.1, 1_000, 2)
    _push_trade(state, 100.2, 2_000, 3)
    market = MarketInfo(
        condition_id="move",
        question="Will BTC go up in 5 minutes?",
        slug="btc-5m-up-down",
        yes_token_id="yes",
        no_token_id="no",
        end_date="2026-04-06T15:00:00Z",
    )

    estimated = estimator.estimate_yes_probability(0.61, market, state)

    assert estimated == 0.61


def test_threshold_market_penalizes_far_above_strike_longshot():
    estimator = MarketProbabilityEstimator(
        strategy_style="momentum",
        vol_window_seconds=300,
        min_sigma=0.001,
    )
    state = RollingState(maxlen=400)
    start = int(datetime(2026, 4, 6, 14, 52, tzinfo=timezone.utc).timestamp() * 1000)
    for index in range(301):
        # Low-volatility tape hovering around 69,654.
        price = 69654.0 + ((index % 5) * 0.08)
        _push_trade(state, price, start + (index * 1_000), index)

    market = MarketInfo(
        condition_id="threshold",
        question="Bitcoin above 69,800 on April 6, 11AM ET?",
        slug="bitcoin-above-69800-on-april-6-2026-11am-et",
        yes_token_id="yes",
        no_token_id="no",
        end_date=(datetime(2026, 4, 6, 15, 0, tzinfo=timezone.utc)).isoformat().replace("+00:00", "Z"),
    )

    estimated = estimator.estimate_yes_probability(0.489, market, state)

    assert estimated < 0.25
    assert estimated < 0.489
