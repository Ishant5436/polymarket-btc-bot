import asyncio

from src.exchange.binance_ws import BinanceWebSocket


def test_transient_connection_error_classification():
    assert BinanceWebSocket._is_transient_connection_error(asyncio.TimeoutError())
    assert BinanceWebSocket._is_transient_connection_error(TimeoutError())
    assert BinanceWebSocket._is_transient_connection_error(
        ConnectionResetError("connection reset")
    )
    assert not BinanceWebSocket._is_transient_connection_error(ValueError("boom"))
