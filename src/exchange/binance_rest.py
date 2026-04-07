"""
Lightweight Binance REST client used for startup warmup/backfill.
"""

import logging
import time

import pandas as pd
import requests

from config.settings import BINANCE

logger = logging.getLogger(__name__)


class BinanceRESTClient:
    """Fetch recent Binance 1-minute klines for live warmup."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "polymarket-btc-bot/0.1",
            }
        )

    def fetch_recent_1m_klines(self, limit: int = 120) -> pd.DataFrame:
        """
        Fetch the most recent closed 1-minute BTCUSDT klines.
        """
        url = f"{BINANCE.rest_base}{BINANCE.kline_endpoint}"
        params = {
            "symbol": BINANCE.symbol,
            "interval": "1m",
            "limit": limit,
        }
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_klines(data)

    @staticmethod
    def _parse_klines(data: list) -> pd.DataFrame:
        """Parse Binance kline payload into the minute-bar shape used by training."""
        if not data:
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "taker_buy_base",
                    "trades_count",
                ]
            )

        columns = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_volume",
            "trades_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        df = pd.DataFrame(data, columns=columns)

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_base",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype(int)
        df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
        df["close_time_ms"] = pd.to_numeric(df["close_time_ms"], errors="coerce")

        # Exclude the currently forming candle so live features only see closed bars.
        now_ms = int(time.time() * 1000)
        df = df[df["close_time_ms"] < now_ms].copy()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "taker_buy_base",
                    "trades_count",
                ]
            )

        df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms")
        df = df[
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "taker_buy_base",
                "trades_count",
            ]
        ].copy()
        df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
