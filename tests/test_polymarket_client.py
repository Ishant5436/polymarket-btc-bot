from types import SimpleNamespace

from src.exchange.polymarket_client import PolymarketClient

TEST_PRIVATE_KEY = "0x" + "1" * 64
TEST_FUNDER_ADDRESS = "0x" + "2" * 40


def test_get_best_bid_ask_supports_dict_books():
    client = PolymarketClient()
    client.get_order_book = lambda token_id: {
        "bids": [{"price": "0.20"}, {"price": "0.49"}, {"price": "0.35"}],
        "asks": [{"price": "0.70"}, {"price": "0.51"}, {"price": "0.80"}],
    }

    best_bid, best_ask = client.get_best_bid_ask("token")

    assert best_bid == 0.49
    assert best_ask == 0.51


def test_get_best_bid_ask_supports_order_book_objects():
    client = PolymarketClient()
    client.get_order_book = lambda token_id: SimpleNamespace(
        bids=[
            SimpleNamespace(price="0.20"),
            SimpleNamespace(price="0.49"),
            SimpleNamespace(price="0.35"),
        ],
        asks=[
            SimpleNamespace(price="0.70"),
            SimpleNamespace(price="0.51"),
            SimpleNamespace(price="0.80"),
        ],
    )

    best_bid, best_ask = client.get_best_bid_ask("token")

    assert best_bid == 0.49
    assert best_ask == 0.51


def test_get_collateral_balance_allowance_parses_string_values():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_balance_allowance = lambda params: {
        "balance": "12.5",
        "allowance": "10.25",
    }

    status = client.get_collateral_balance_allowance()

    assert status is not None
    assert status.balance == 12.5
    assert status.allowance == 10.25
    assert status.available_to_trade == 10.25


def test_get_collateral_balance_allowance_normalizes_integer_base_units():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_balance_allowance = lambda params: {
        "balance": "1079239",
        "allowance": "500000",
    }

    status = client.get_collateral_balance_allowance()

    assert status is not None
    assert status.balance == 1.079239
    assert status.allowance == 0.5
    assert status.available_to_trade == 0.5


def test_get_collateral_balance_allowance_parses_allowances_map():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_balance_allowance = lambda params: {
        "balance": "3000000",
        "allowances": {
            "0xexchange": "0",
            "0xneg-risk": "2500000",
        },
    }

    status = client.get_collateral_balance_allowance()

    assert status is not None
    assert status.balance == 3.0
    assert status.allowance == 2.5
    assert status.available_to_trade == 2.5
    assert status.allowances_by_spender == {
        "0xexchange": 0.0,
        "0xneg-risk": 2.5,
    }


def test_has_sufficient_collateral_fails_when_allowance_is_too_low():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_balance_allowance = lambda params: {
        "balance": "50.0",
        "allowance": "0.40",
    }

    assert client.has_sufficient_collateral(0.50) is False


def test_get_available_collateral_returns_allowance_capped_balance():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_balance_allowance = lambda params: {
        "balance": "5.0",
        "allowance": "3.25",
    }

    available = client.get_available_collateral()

    assert available == 3.25


def test_get_trade_history_slices_client_results_without_invalid_limit_kwarg():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        api_key="test-api-key",
        api_secret="test-api-secret",
        api_passphrase="test-api-passphrase",
    )
    client._client.get_trades = lambda: [
        {"id": "3"},
        {"id": "2"},
        {"id": "1"},
    ]

    trades = client.get_trade_history(limit=2)

    assert trades == [{"id": "3"}, {"id": "2"}]


def test_tracking_address_prefers_funder_address():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        funder_address=TEST_FUNDER_ADDRESS,
    )

    assert client.tracking_address == TEST_FUNDER_ADDRESS


def test_get_closed_positions_pages_results_from_data_api():
    client = PolymarketClient(
        private_key=TEST_PRIVATE_KEY,
        funder_address=TEST_FUNDER_ADDRESS,
    )

    requested_params = []
    payloads = iter(
        [
            [
                {"asset": "asset-1", "realizedPnl": 0.10},
                {"asset": "asset-2", "realizedPnl": -0.05},
            ],
            [
                {"asset": "asset-3", "realizedPnl": 0.02},
            ],
        ]
    )

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params, timeout):
        requested_params.append((url, params, timeout))
        return DummyResponse(next(payloads))

    client._data_session.get = fake_get

    positions = client.get_closed_positions(limit=2, max_records=5)

    assert positions == [
        {"asset": "asset-1", "realizedPnl": 0.10},
        {"asset": "asset-2", "realizedPnl": -0.05},
        {"asset": "asset-3", "realizedPnl": 0.02},
    ]
    assert requested_params == [
        (
            "https://data-api.polymarket.com/closed-positions",
            {"user": TEST_FUNDER_ADDRESS, "limit": 2, "offset": 0},
            10,
        ),
        (
            "https://data-api.polymarket.com/closed-positions",
            {"user": TEST_FUNDER_ADDRESS, "limit": 2, "offset": 2},
            10,
        ),
    ]
