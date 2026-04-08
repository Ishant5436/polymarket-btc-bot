from unittest.mock import AsyncMock

from websockets.protocol import State

from src.exchange.polymarket_ws import PolymarketWebSocket


def test_subscribe_sends_when_connection_is_open():
    ws = PolymarketWebSocket()
    ws._ws = AsyncMock()
    ws._ws.state = State.OPEN

    import asyncio

    asyncio.run(ws.subscribe(["asset-1", "asset-2"]))

    ws._ws.send.assert_awaited_once()
    assert ws.get_book("asset-1") == {"bids": [], "asks": []}
    assert ws.get_book("asset-2") == {"bids": [], "asks": []}


def test_subscribe_tracks_assets_without_sending_when_connection_is_not_open():
    ws = PolymarketWebSocket()
    ws._ws = AsyncMock()
    ws._ws.state = State.CLOSED

    import asyncio

    asyncio.run(ws.subscribe(["asset-1"]))

    ws._ws.send.assert_not_awaited()
    assert "asset-1" in ws._subscribed_assets
    assert ws.get_book("asset-1") == {"bids": [], "asks": []}


def test_connection_is_open_uses_websockets_state_enum():
    ws = PolymarketWebSocket()
    ws._ws = AsyncMock()
    ws._ws.state = State.OPEN
    assert ws._connection_is_open() is True

    ws._ws.state = State.CLOSING
    assert ws._connection_is_open() is False
