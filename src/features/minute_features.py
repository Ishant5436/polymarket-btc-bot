"""
Shared minute-bar feature engineering used by both training and live inference.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import FEATURES
from src.features.indicators import (
    fractional_differentiation,
    micro_price_momentum,
    order_book_imbalance,
    rolling_hurst_exponent,
    rolling_volatility_fast,
    vwap_deviation_fast,
)

BAR_INPUT_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_base",
    "trades_count",
    "liq_long_notional",
    "liq_short_notional",
]


def _compute_pseudo_liquidation_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Estimate liquidation-like pressure directly from spot bars.

    The signal is strongest when a bar combines:
    - outsized localized volume versus its recent baseline,
    - a large intrabar spread,
    - strongly one-sided taker flow,
    - a close that confirms the directional move.
    """
    open_px = pd.to_numeric(df["open"], errors="coerce")
    high_px = pd.to_numeric(df["high"], errors="coerce")
    low_px = pd.to_numeric(df["low"], errors="coerce")
    close_px = pd.to_numeric(df["close"], errors="coerce")

    total_vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    taker_buy = (
        pd.to_numeric(df["taker_buy_base"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    taker_buy = taker_buy.clip(upper=total_vol)
    taker_sell = (total_vol - taker_buy).clip(lower=0.0)

    bar_range = (high_px.fillna(0.0) - low_px.fillna(0.0)).clip(lower=0.0)
    range_safe = bar_range.replace(0.0, np.nan)
    price_ref = close_px.replace(0.0, np.nan)
    spread_pct = (bar_range / price_ref).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volume_baseline = total_vol.rolling(window=30, min_periods=1).median().replace(0.0, np.nan)
    volume_shock = (total_vol / volume_baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    volume_multiplier = 1.0 + 0.75 * (volume_shock - 1.0).clip(lower=0.0, upper=4.0)

    flow_imbalance = ((taker_buy - taker_sell) / total_vol.replace(0.0, np.nan)).fillna(0.0)
    sell_dominance = (-flow_imbalance).clip(lower=0.0, upper=1.0)
    buy_dominance = flow_imbalance.clip(lower=0.0, upper=1.0)

    close_location = (
        ((close_px - low_px) - (high_px - close_px))
        / range_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bearish_close = (-close_location).clip(lower=0.0, upper=1.0)
    bullish_close = close_location.clip(lower=0.0, upper=1.0)

    bearish_body = (
        (open_px - close_px)
        / range_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    bullish_body = (
        (close_px - open_px)
        / range_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)

    directional_confirmation_long = 0.5 * bearish_body + 0.5 * bearish_close
    directional_confirmation_short = 0.5 * bullish_body + 0.5 * bullish_close

    pseudo_liq_long = np.where(
        close_px < open_px,
        taker_sell * spread_pct * volume_multiplier * sell_dominance * directional_confirmation_long,
        0.0,
    )
    pseudo_liq_short = np.where(
        close_px > open_px,
        taker_buy * spread_pct * volume_multiplier * buy_dominance * directional_confirmation_short,
        0.0,
    )

    return pd.Series(pseudo_liq_long - pseudo_liq_short, index=df.index, dtype=np.float64)


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the training/live feature frame from 1-minute BTC bars.
    """
    if df.empty:
        return df.copy()

    df = df.sort_values("open_time").reset_index(drop=True).copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()

    bid_vol = df["taker_buy_base"].to_numpy(dtype=np.float64)
    ask_vol = (df["volume"] - df["taker_buy_base"]).to_numpy(dtype=np.float64)
    df["obi"] = order_book_imbalance(bid_vol, ask_vol)

    close = df["close"].to_numpy(dtype=np.float64)
    df["momentum_1m"] = micro_price_momentum(close, window=1)
    df["momentum_2m"] = micro_price_momentum(close, window=2)
    df["momentum_5m"] = micro_price_momentum(close, window=5)
    df["momentum_10m"] = micro_price_momentum(close, window=10)

    df["pseudo_liq_imbalance_1m"] = _compute_pseudo_liquidation_imbalance(df)
    df["pseudo_liq_imbalance_5m"] = df["pseudo_liq_imbalance_1m"].rolling(window=5, min_periods=1).sum()

    log_rets = df["log_return"].to_numpy(dtype=np.float64)
    df["hurst"] = rolling_hurst_exponent(
        log_rets,
        window=FEATURES.hurst_window,
        max_lag=FEATURES.hurst_max_lag,
    )

    df["fracdiff_close"] = fractional_differentiation(
        close,
        d=FEATURES.fracdiff_d,
    )

    df["vol_1m"] = rolling_volatility_fast(df["log_return"], window=3)  # Bug D fix: window=3 (distinct from vol_5m=5)
    df["vol_5m"] = rolling_volatility_fast(df["log_return"], window=5)
    df["vol_30m"] = rolling_volatility_fast(df["log_return"], window=30)
    df["vol_60m"] = rolling_volatility_fast(df["log_return"], window=60)

    df["vwap_dev_5m"] = vwap_deviation_fast(df["close"], df["volume"], window=5)
    df["vwap_dev_30m"] = vwap_deviation_fast(df["close"], df["volume"], window=30)

    total_vol = df["volume"]
    buy_ratio = df["taker_buy_base"] / total_vol.replace(0, np.nan)
    df["trade_flow_imb"] = 2 * buy_ratio - 1
    df["trade_flow_imb_5m"] = (
        df["trade_flow_imb"].rolling(window=5, min_periods=1).mean()
    )
    df["trade_flow_imb_30m"] = (
        df["trade_flow_imb"].rolling(window=30, min_periods=1).mean()
    )

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol_ratio_5m"] = (
        df["volume"] / df["volume"].rolling(window=5, min_periods=1).mean()
    )
    df["vol_ratio_30m"] = (
        df["volume"] / df["volume"].rolling(window=30, min_periods=1).mean()
    )
    df["trades_intensity"] = (
        df["trades_count"]
        / df["trades_count"].rolling(window=30, min_periods=1).mean()
    )

    # Improvement 8: Multi-timeframe trend alignment
    # +3/-3 = all timeframes agree, 0 = mixed signals
    df["trend_alignment"] = (
        np.sign(df["momentum_1m"].fillna(0))
        + np.sign(df["momentum_5m"].fillna(0))
        + np.sign(df["momentum_10m"].fillna(0))
    )

    return df


def aggregate_trades_to_1m_bars(
    trades: Iterable[object],
    liquidations: Iterable[object] = (),
    include_incomplete_last_bar: bool = False,
) -> pd.DataFrame:
    """
    Aggregate tick trades into synthetic 1-minute bars.

    The live model was trained on minute bars, so we drop the trailing partial
    minute by default and only emit fully-formed bars.
    """
    rows = [
        {
            "timestamp": int(trade.timestamp),
            "price": float(trade.price),
            "quantity": float(trade.quantity),
            "is_buyer_maker": bool(trade.is_buyer_maker),
        }
        for trade in trades
    ]
    if not rows:
        return pd.DataFrame(columns=BAR_INPUT_COLUMNS)

    df = pd.DataFrame.from_records(rows)
    df["bucket_start_ms"] = (df["timestamp"] // 60000) * 60000

    latest_bucket_start = int(df["bucket_start_ms"].iloc[-1])
    latest_bucket_age_ms = int(df["timestamp"].iloc[-1]) - latest_bucket_start
    if not include_incomplete_last_bar and latest_bucket_age_ms < 59000:
        df = df[df["bucket_start_ms"] < latest_bucket_start]

    if df.empty:
        return pd.DataFrame(columns=BAR_INPUT_COLUMNS)

    df["taker_buy_base"] = np.where(~df["is_buyer_maker"], df["quantity"], 0.0)
    bars = (
        df.groupby("bucket_start_ms", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            taker_buy_base=("taker_buy_base", "sum"),
            trades_count=("price", "size"),
        )
        .reset_index()
    )

    bars.insert(0, "open_time", pd.to_datetime(bars["bucket_start_ms"], unit="ms"))
    bars = bars.drop(columns=["bucket_start_ms"])
    
    # Process liquidations
    liq_rows = [
        {
            "bucket_start_ms": (int(liq.timestamp) // 60000) * 60000,
            "side": liq.side,
            "notional": float(liq.price) * float(liq.quantity)
        }
        for liq in liquidations
    ]
    
    if liq_rows:
        liq_df = pd.DataFrame.from_records(liq_rows)
        # Liquidated longs = SELL side order forced. Liquidated shorts = BUY side forced.
        liq_df["is_long"] = liq_df["side"] == "SELL"
        liq_df["is_short"] = liq_df["side"] == "BUY"
        
        liq_agg = liq_df.groupby("bucket_start_ms").apply(
            lambda x: pd.Series({
                "liq_long_notional": x.loc[x["is_long"], "notional"].sum(),
                "liq_short_notional": x.loc[x["is_short"], "notional"].sum(),
            })
        ).reset_index()
        
        bars["bucket_start_ms_tmp"] = (
            bars["open_time"]
            .to_numpy(dtype="datetime64[ms]")
            .astype(np.int64)
        )
        bars = pd.merge(bars, liq_agg, left_on="bucket_start_ms_tmp", right_on="bucket_start_ms", how="left")
        bars.fillna({"liq_long_notional": 0.0, "liq_short_notional": 0.0}, inplace=True)
        bars.drop(columns=["bucket_start_ms", "bucket_start_ms_tmp"], inplace=True)
    else:
        bars["liq_long_notional"] = 0.0
        bars["liq_short_notional"] = 0.0

    return bars[BAR_INPUT_COLUMNS]
