"""
Polymarket WebSocket client for ultra-low latency order book streaming.
Replaces REST polling with instantaneous push-based updates.
"""

import asyncio
import json
import logging
from typing import Any, Optional

import websockets
from websockets.protocol import State

from config.settings import POLYMARKET

logger = logging.getLogger(__name__)


class PolymarketWebSocket:
    """
    Subscribes to the Polymarket CLOB WebSocket to maintain a real-time,
    in-memory L2 limit order book for active tokens.
    """

    def __init__(self, ws_url: str = POLYMARKET.ws_url):
        self._ws_url = ws_url
        self._running = False
        self._ws: Optional[Any] = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        
        # In-memory L2 Books
        # Dict[asset_id, Dict[str, list[dict]]]
        # e.g., {"123...": {"bids": [{"price": "0.45", "size": "100"}...], "asks": [...]}}
        self._books: dict[str, dict[str, Any]] = {}
        
        self._subscribed_assets: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self):
        """Start the WebSocket connection and message loop."""
        self._running = True
        logger.info("Polymarket WS starting | url=%s", self._ws_url)
        
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                self._running = False
                logger.info("Polymarket WS task cancelled")
                break
            except Exception as e:
                logger.error("Polymarket WS error: %s", e)
                
            if self._running:
                logger.warning("Polymarket WS reconnecting in %.1fs", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._max_reconnect_delay, self._reconnect_delay * 2)

    async def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Polymarket WS stopped")

    async def subscribe(self, asset_ids: list[str]):
        """
        Subscribe to book updates for specific asset IDs.
        """
        new_assets = [a for a in asset_ids if a not in self._subscribed_assets]
        if not new_assets:
            return
            
        for a in new_assets:
            self._subscribed_assets.add(a)
            if a not in self._books:
                self._books[a] = {"bids": [], "asks": []}
            
        if self._connection_is_open():
            payload = {
                "assets_ids": new_assets,
                "type": "market"
            }
            try:
                await self._ws.send(json.dumps(payload))
                logger.info("Polymarket WS subscribed | assets=%d", len(new_assets))
            except Exception as e:
                logger.error("Polymarket WS subscribe error: %s", e)

    async def _connect_and_listen(self):
        """Establish connection and process incoming messages."""
        async with websockets.connect(self._ws_url) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connect
            logger.info("Polymarket WS connected")
            
            # Re-subscribe to existing assets on reconnect
            if self._subscribed_assets:
                cached_assets = list(self._subscribed_assets)
                self._subscribed_assets.clear()
                await self.subscribe(cached_assets)
            
            async for message in ws:
                if not self._running:
                    break
                await self._handle_message(message)
        if self._ws is ws:
            self._ws = None

    async def _handle_message(self, raw_message: str):
        """Parse and route incoming WS messages."""
        try:
            data = json.loads(raw_message)
            
            if isinstance(data, list):
                # Sometimes events arrive in a list
                for item in data:
                    await self._process_event(item)
            else:
                await self._process_event(data)
                
        except json.JSONDecodeError:
            logger.error("Failed to parse Polymarket WS JSON: %s", raw_message[:100])
        except Exception as e:
            logger.error("Error handling Polymarket WS message: %s", e)

    async def _process_event(self, event: dict):
        """Update local book using book events."""
        event_type = event.get("event_type")
        asset_id = event.get("asset_id")
        
        if event_type == "book" and asset_id:
            bids = event.get("bids", [])
            asks = event.get("asks", [])
            
            # Polymarket CLOB WS typically sends full book state per event
            self._books[asset_id] = {
                "bids": bids,
                "asks": asks,
                "timestamp": event.get("timestamp")
            }

    def get_book(self, asset_id: str) -> dict:
        """
        Get a copy of the current L2 book for an asset (synchronous).
        """
        book = self._books.get(asset_id, {"bids": [], "asks": []})
        return {
            "bids": list(book.get("bids", [])),
            "asks": list(book.get("asks", [])),
        }

    def get_best_bid_ask(self, asset_id: str) -> tuple[Optional[float], Optional[float]]:
        """
        Get the current best (highest) bid and best (lowest) ask (synchronous).
        """
        book = self._books.get(asset_id, {})
        bids = book.get("bids", [])
        asks = book.get("asks", [])
            
        best_bid = None
        best_ask = None
        
        if bids:
            best_bid = max(float(level.get("price", 0)) for level in bids)
        if asks:
            best_ask = min(float(level.get("price", 0)) for level in asks)
            
        return best_bid, best_ask

    def _connection_is_open(self) -> bool:
        """Return True when the current websocket connection can accept sends."""
        return bool(self._ws and getattr(self._ws, "state", None) == State.OPEN)
