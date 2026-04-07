"""
Binance WebSocket consumer for real-time BTC/USDT aggregate trade data.
Maintains a persistent async connection with automatic reconnection.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from config.settings import BINANCE

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """
    Persistent async WebSocket consumer for the Binance btcusdt@aggTrade stream.
    Pushes parsed trade data into an asyncio.Queue for downstream consumption.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        output_queue: Optional[asyncio.Queue] = None,
    ):
        self._url = url or BINANCE.ws_url
        self._queue = output_queue or asyncio.Queue(maxsize=10_000)
        self._running = False
        self._reconnect_delay = BINANCE.reconnect_delay_base
        self._connection_start: Optional[float] = None
        self._message_count = 0
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    @property
    def queue(self) -> asyncio.Queue:
        """The output queue where parsed trade data is pushed."""
        return self._queue

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self):
        """
        Start the WebSocket consumer loop.
        Automatically reconnects on disconnection with exponential backoff.
        Rotates connection before the 24-hour Binance limit.
        """
        self._running = True
        logger.info("Binance WS starting | url=%s", self._url)

        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                logger.info("Binance WS cancelled")
                break
            except Exception as e:
                if self._is_transient_connection_error(e):
                    logger.warning("Binance WS transient error: %s", e)
                else:
                    logger.error("Binance WS error: %s", e)

            if self._running:
                logger.info("Reconnecting in %.1fs...", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff with cap
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    BINANCE.reconnect_delay_max,
                )

    async def _connect_and_consume(self):
        """Establish connection and consume messages."""
        self._connection_start = time.time()
        self._message_count = 0
        self._reconnect_delay = BINANCE.reconnect_delay_base  # Reset on successful connect

        async with websockets.connect(
            self._url,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            logger.info("Binance WS connected")

            try:
                while self._running:
                    # Check connection lifetime (rotate before 24h)
                    elapsed_hours = (time.time() - self._connection_start) / 3600
                    if elapsed_hours >= BINANCE.connection_lifetime_hours:
                        logger.info(
                            "Connection lifetime reached (%.1fh), rotating...",
                            elapsed_hours,
                        )
                        break

                    # Receive with timeout
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        # No data for 30s — connection may be stale
                        logger.warning("No data for 30s, checking connection...")
                        continue

                    # Parse the trade
                    trade = self._parse_agg_trade(raw)
                    if trade:
                        # Non-blocking put — drop oldest if queue is full
                        try:
                            self._queue.put_nowait(trade)
                        except asyncio.QueueFull:
                            # Drain one and push
                            try:
                                self._queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self._queue.put_nowait(trade)

                        self._message_count += 1
                        if self._message_count % 10_000 == 0:
                            logger.debug(
                                "Processed %d trades | queue_size=%d",
                                self._message_count,
                                self._queue.qsize(),
                            )

            except ConnectionClosed as e:
                logger.warning("Binance WS closed: code=%s reason=%s", e.code, e.reason)
            finally:
                self._ws = None

    @staticmethod
    def _is_transient_connection_error(exc: Exception) -> bool:
        """Classify expected network churn so reconnect logs stay proportional."""
        return isinstance(exc, (asyncio.TimeoutError, TimeoutError, OSError))

    @staticmethod
    def _parse_agg_trade(raw: str) -> Optional[dict]:
        """
        Parse a Binance aggTrade message.
        
        Raw format:
        {
            "e": "aggTrade",
            "E": 1672515782136,    # Event time
            "s": "BTCUSDT",         # Symbol
            "a": 123456789,         # Aggregate trade ID
            "p": "96543.21",        # Price
            "q": "0.00100",         # Quantity
            "f": 100,               # First trade ID
            "l": 105,               # Last trade ID
            "T": 1672515782136,     # Trade time
            "m": false,             # Is buyer maker?
            "M": true               # Is best match?
        }
        """
        try:
            data = json.loads(raw)
            return {
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "timestamp": data["T"],       # ms since epoch
                "is_buyer_maker": data["m"],   # True = sell aggressor
                "trade_id": data["a"],
                "event_time": data["E"],
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Failed to parse trade: %s", e)
            return None

    async def stop(self):
        """Gracefully stop the WebSocket consumer."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info(
            "Binance WS stopped | total_messages=%d",
            self._message_count,
        )
