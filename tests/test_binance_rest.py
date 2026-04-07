import time

from src.exchange.binance_rest import BinanceRESTClient


def test_parse_klines_drops_incomplete_last_bar():
    now_ms = int(time.time() * 1000)
    current_open_ms = (now_ms // 60_000) * 60_000
    previous_open_ms = current_open_ms - 60_000

    payload = [
        [
            previous_open_ms,
            "100",
            "105",
            "99",
            "104",
            "10",
            previous_open_ms + 59_999,
            "0",
            "12",
            "6",
            "0",
            "0",
        ],
        [
            current_open_ms,
            "104",
            "106",
            "103",
            "105",
            "11",
            current_open_ms + 59_999,
            "0",
            "13",
            "7",
            "0",
            "0",
        ],
    ]

    parsed = BinanceRESTClient._parse_klines(payload)

    assert len(parsed) == 1
    assert float(parsed.iloc[0]["close"]) == 104.0
    assert int(parsed.iloc[0]["trades_count"]) == 12
    assert float(parsed.iloc[0]["taker_buy_base"]) == 6.0
