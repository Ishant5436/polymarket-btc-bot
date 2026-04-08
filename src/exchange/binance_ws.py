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
    
    Supports automatic fallback from Futures (fstream) to Spot (stream) endpoints
    if the primary connection fails. Spot streams provide aggTrade data but lack
    forceOrder liquidation events.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        url_spot_fallback: Optional[str] = None,
        output_queue: Optional[asyncio.Queue] = None,
        enable_spot_fallback: bool = True,
    ):
        self._url_primary = url or BINANCE.ws_url
        self._url_spot_fallback = url_spot_fallback or BINANCE.ws_url_spot_fallback
        self._queue = output_queue or asyncio.Queue(maxsize=10_000)
        self._running = False
        self._reconnect_delay = BINANCE.reconnect_delay_base
        self._connection_start: Optional[float] = None
        self._message_count = 0
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._enable_spot_fallback = enable_spot_fallback
        self._current_url = self._url_primary
        self._fallback_active = False

    @property
    def queue(self) -> asyncio.Queue:
        """The output queue where parsed trade data is pushed."""
        return self._queue

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def using_fallback(self) -> bool:
        """Returns True if currently using Spot fallback stream instead of Futures."""
        return self._fallback_active

    @property
    def stream_type(self) -> str:
        """Returns 'Spot' if using fallback, 'Futures' if using primary stream."""
        return "Spot" if self._fallback_active else "Futures"

    async def start(self):
        """
        Start the WebSocket consumer loop.
        Automatically reconnects on disconnection with exponential backoff.
        Rotates connection before the 24-hour Binance limit.
        Falls back to Spot stream if Futures endpoint fails (if enabled).
        """
        self._running = True
        logger.info("Binance WS starting | url=%s", self._current_url)

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

                # Try Spot fallback on non-transient errors if enabled and not already active
                if (
                    self._enable_spot_fallback
                    and not self._fallback_active
                    and not self._is_transient_connection_error(e)
                ):
                    logger.warning(
                        "Futures endpoint failed (%s). Attempting fallback to Spot stream...",
                        type(e).__name__,
                    )
                    self._current_url = self._url_spot_fallback
                    self._fallback_active = True
                    self._reconnect_delay = BINANCE.reconnect_delay_base  # Reset backoff
                    await asyncio.sleep(1)  # Brief delay before retry with fallback
                    continue

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

        stream_type = "Spot" if self._fallback_active else "Futures"
        logger.info("Connecting to %s stream: %s", stream_type, self._current_url)

        async with websockets.connect(
            self._current_url,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            logger.info("Binance WS connected (%s stream)", stream_type)

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

                    # Parse multiplexed stream or raw message
                    try:
                        data = json.loads(raw)
                        stream = data.get("stream")
                        payload = data.get("data")
                        
                        if not stream or not payload:
                            # Raw (non-multiplexed) payload parsing fallback
                            payload = data

                        event_type = payload.get("e")
                        if event_type == "aggTrade":
                            # Parse the trade
                            trade = self._parse_agg_trade(payload)
                            if trade:
                                self._enqueue_message(trade)
                        elif event_type == "forceOrder":
                            # Parse the liquidation
                            liq = self._parse_force_order(payload)
                            if liq:
                                self._enqueue_message(liq)
                    except json.JSONDecodeError:
                        logger.debug("Failed to decode JSON from stream")

            except ConnectionClosed as e:
                logger.warning("Binance WS closed: code=%s reason=%s", e.code, e.reason)
            finally:
                self._ws = None

    def _enqueue_message(self, message: dict):
        """Put a message in the queue, evicting the oldest if full."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(message)

        self._message_count += 1
        if self._message_count % 10_000 == 0:
            logger.debug(
                "Processed %d socket events | queue_size=%d",
                self._message_count,
                self._queue.qsize(),
            )

    @staticmethod
    def _is_transient_connection_error(exc: Exception) -> bool:
        """Classify expected network churn so reconnect logs stay proportional."""
        return isinstance(exc, (asyncio.TimeoutError, TimeoutError, OSError))

    @staticmethod
    def _parse_agg_trade(data: dict) -> Optional[dict]:
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
            return {
                "type": "trade",
                "price": float(data["p"]),
                "quantity": float(data["q"]),
                "timestamp": data["T"],       # ms since epoch
                "is_buyer_maker": data["m"],   # True = sell aggressor
                "trade_id": data["a"],
                "event_time": data["E"],
            }
        except (KeyError, ValueError) as e:
            logger.debug("Failed to parse trade: %s", e)
            return None

    @staticmethod
    def _parse_force_order(data: dict) -> Optional[dict]:
        """
        Parse a Binance Futures forceOrder (Liquidation) message.
        """
        try:
            o = data["o"]
            return {
                "type": "liquidation",
                "side": o["S"],         # "BUY" or "SELL"
                "price": float(o["p"]),
                "quantity": float(o["q"]),
                "timestamp": o["T"],    # ms since epoch
                "event_time": data["E"],
            }
        except (KeyError, ValueError) as e:
            logger.debug("Failed to parse liquidation: %s", e)
            return None

    async def stop(self):
        """Gracefully stop the WebSocket consumer."""
        self._running = False
        if self._ws:
            await self._ws.close()
        stream_type = "Spot" if self._fallback_active else "Futures"
        logger.info(
            "Binance WS stopped | stream=%s | total_messages=%d",
            stream_type,
            self._message_count,
        )
