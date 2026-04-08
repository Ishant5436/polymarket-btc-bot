#!/usr/bin/env python3
"""
Diagnostic script to test Binance WebSocket connectivity.
Tests both Futures (fstream) and Spot (stream) endpoints to identify ISP/network blocks.
"""

import asyncio
import logging
import time
from typing import Tuple

import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def test_endpoint(url: str, name: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Attempt to connect to a WebSocket endpoint and receive one message.
    
    Returns:
        (success: bool, status: str)
    """
    try:
        logger.info(f"Testing {name}: {url}")
        try:
            async with websockets.connect(
                url,
                ping_interval=5,
                ping_timeout=5,
            ) as ws:
                logger.info(f"✓ Connected to {name}")
                
                # Try to receive one message with a short timeout
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3)
                    logger.info(f"✓ Received data from {name} ({len(msg)} bytes)")
                    return (True, "Connected and receiving data")
                except asyncio.TimeoutError:
                    logger.warning(f"✓ Connected but no data from {name} after 3s (expected if no active trades)")
                    return (True, "Connected but no data received (timeout)")
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Connection timeout to {name}")
                    
    except asyncio.TimeoutError:
        msg = f"Connection timeout to {name} (likely ISP block or network issue)"
        logger.error(f"✗ {msg}")
        return (False, msg)
    except ConnectionRefusedError as e:
        msg = f"Connection refused to {name} (endpoint unreachable)"
        logger.error(f"✗ {msg}: {e}")
        return (False, msg)
    except OSError as e:
        msg = f"Network error to {name}: {e}"
        logger.error(f"✗ {msg}")
        return (False, msg)
    except Exception as e:
        msg = f"Unexpected error connecting to {name}: {type(e).__name__}: {e}"
        logger.error(f"✗ {msg}")
        return (False, msg)


async def run_diagnostics():
    """Run full connectivity diagnostics."""
    logger.info("=" * 80)
    logger.info("BINANCE WEBSOCKET CONNECTIVITY DIAGNOSTIC")
    logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 80)
    
    # Test configurations
    tests = [
        {
            "name": "Futures Multiplexed (Current)",
            "url": "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@forceOrder",
        },
        {
            "name": "Futures Single Stream",
            "url": "wss://fstream.binance.com/ws/btcusdt@aggTrade",
        },
        {
            "name": "Spot Multiplexed (Fallback)",
            "url": "wss://stream.binance.com:9443/ws?streams=btcusdt@aggTrade",
        },
        {
            "name": "Spot Single Stream (Fallback)",
            "url": "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
        },
        {
            "name": "Spot Alt Port (Fallback)",
            "url": "wss://stream.binance.com/ws/btcusdt@aggTrade",
        },
    ]
    
    results = {}
    
    for test in tests:
        logger.info(f"\n[Test {len(results) + 1}/{len(tests)}]")
        success, status = await test_endpoint(test["url"], test["name"])
        results[test["name"]] = (success, status)
        await asyncio.sleep(1)  # Rate limiting between tests
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for name, (success, status) in results.items():
        status_icon = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status_icon:8} | {name:45} | {status}")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    futures_working = any(s for n, (s, _) in results.items() if "Futures" in n)
    spot_working = any(s for n, (s, _) in results.items() if "Spot" in n)
    
    if futures_working and spot_working:
        logger.info("✓ Both Futures and Spot endpoints are reachable.")
        logger.info("  No ISP blocks detected. Issue likely elsewhere (auth, firewall, etc.)")
    elif futures_working and not spot_working:
        logger.error("✗ Spot endpoints are blocked but Futures work.")
        logger.error("  (Unusual pattern; may indicate regional restriction)")
    elif not futures_working and spot_working:
        logger.warning("⚠ Futures endpoints are blocked; Spot is reachable.")
        logger.warning("  This is the expected ISP block pattern in India.")
        logger.warning("  RECOMMENDED: Use Spot fallback strategy (wss://stream.binance.com:9443/ws)")
    else:
        logger.critical("✗ Both Futures and Spot endpoints are unreachable!")
        logger.critical("  This suggests a broader network/firewall issue or complete ISP block.")
        logger.critical("  Next steps: Check VPN, proxy, or firewall rules.")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
