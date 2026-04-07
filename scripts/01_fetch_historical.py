#!/usr/bin/env python3
"""
01_fetch_historical.py
Pulls historical BTC/USDT minute-level kline data and aggregated trades
from the Binance REST API. Supports pagination and rate limiting.

Usage:
    python scripts/01_fetch_historical.py [--days 90] [--trades]
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BINANCE, PATHS


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 90,
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """
    Fetch minute-level kline (candlestick) data from Binance REST API.
    Paginates backwards from now, respecting rate limits.
    """
    url = f"{BINANCE.rest_base}{BINANCE.kline_endpoint}"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_time

    print(f"[*] Fetching {days} days of {interval} klines for {symbol}...")
    print(f"    From: {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)}")
    print(f"    To:   {datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)}")

    request_count = 0
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit_per_request,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[!] Request error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        all_klines.extend(data)
        request_count += 1

        # Move start to after the last candle received
        current_start = data[-1][0] + 1  # close_time + 1ms

        if request_count % 10 == 0:
            n_minutes = len(all_klines)
            pct = min(100, (current_start - start_time) / (end_time - start_time) * 100)
            print(f"    [{pct:5.1f}%] Fetched {n_minutes:,} candles ({request_count} requests)")

        # Binance rate limit: 1200 requests/min for general endpoints
        # Stay well within limits
        time.sleep(0.1)

    print(f"[✓] Fetched {len(all_klines):,} klines in {request_count} requests.")

    # Build DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_klines, columns=columns)

    # Type conversions
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["trades_count"] = df["trades_count"].astype(int)
    df.drop(columns=["ignore"], inplace=True)

    # Deduplicate by open_time
    df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_agg_trades(
    symbol: str = "BTCUSDT",
    days: int = 7,
    limit_per_request: int = 1000,
    max_records: int = 500_000,
) -> pd.DataFrame:
    """
    Fetch aggregated trade data from Binance REST API.
    Limited to recent `days` to keep dataset manageable.
    """
    url = f"{BINANCE.rest_base}{BINANCE.agg_trades_endpoint}"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_trades = []
    current_start = start_time

    print(f"\n[*] Fetching {days} days of aggTrades for {symbol}...")
    print(f"    Max records: {max_records:,}")

    request_count = 0
    while current_start < end_time and len(all_trades) < max_records:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit_per_request,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[!] Request error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        all_trades.extend(data)
        request_count += 1
        current_start = data[-1]["T"] + 1  # timestamp + 1ms

        if request_count % 50 == 0:
            print(f"    Fetched {len(all_trades):,} trades ({request_count} requests)")

        time.sleep(0.05)  # Lighter rate limiting for trade endpoint

    print(f"[✓] Fetched {len(all_trades):,} aggTrades in {request_count} requests.")

    df = pd.DataFrame(all_trades)
    df.rename(columns={
        "a": "agg_trade_id",
        "p": "price",
        "q": "quantity",
        "f": "first_trade_id",
        "l": "last_trade_id",
        "T": "timestamp",
        "m": "is_buyer_maker",
        "M": "is_best_match",
    }, inplace=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    df.drop_duplicates(subset=["agg_trade_id"], keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch historical BTC/USDT data from Binance")
    parser.add_argument("--days", type=int, default=90, help="Days of kline history (default: 90)")
    parser.add_argument("--trade-days", type=int, default=7, help="Days of aggTrade history (default: 7)")
    parser.add_argument("--trades", action="store_true", help="Also fetch aggregated trades")
    parser.add_argument("--max-trades", type=int, default=500_000, help="Max aggTrade records")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(PATHS.raw_data_dir, exist_ok=True)

    # --- Klines ---
    klines_df = fetch_klines(symbol=BINANCE.symbol, days=args.days)
    klines_df.to_csv(PATHS.klines_path, index=False)
    print(f"[✓] Saved klines to {PATHS.klines_path}")
    print(f"    Shape: {klines_df.shape}")
    print(f"    Date range: {klines_df['open_time'].iloc[0]} → {klines_df['open_time'].iloc[-1]}")

    # --- AggTrades (optional) ---
    if args.trades:
        trades_df = fetch_agg_trades(
            symbol=BINANCE.symbol,
            days=args.trade_days,
            max_records=args.max_trades,
        )
        trades_df.to_csv(PATHS.agg_trades_path, index=False)
        print(f"[✓] Saved aggTrades to {PATHS.agg_trades_path}")
        print(f"    Shape: {trades_df.shape}")
        print(f"    Date range: {trades_df['timestamp'].iloc[0]} → {trades_df['timestamp'].iloc[-1]}")

    print("\n[✓] Data fetch complete.")


if __name__ == "__main__":
    main()
