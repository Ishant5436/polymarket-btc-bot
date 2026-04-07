"""
Feature indicators for BTC price prediction.
Used both in offline training (batch) and live inference (real-time).
"""

import numpy as np
import pandas as pd


def order_book_imbalance(bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
    """
    Compute order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
    Returns values in [-1, 1]. Positive = buy pressure, Negative = sell pressure.
    
    For historical data where we lack true L2 book data, we proxy using
    taker_buy_volume vs total_volume from kline data.
    """
    total = bid_volume + ask_volume
    # Avoid division by zero
    mask = total > 0
    result = np.zeros_like(bid_volume, dtype=np.float64)
    result[mask] = (bid_volume[mask] - ask_volume[mask]) / total[mask]
    return result


def micro_price_momentum(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Log return over a rolling window of `window` periods.
    momentum_t = ln(price_t / price_{t-window})
    """
    result = np.full_like(prices, np.nan, dtype=np.float64)
    valid = (prices[window:] > 0) & (prices[:-window] > 0)
    result[window:] = np.where(
        valid,
        np.log(prices[window:] / prices[:-window]),
        np.nan,
    )
    return result


def rolling_hurst_exponent(
    log_returns: np.ndarray, window: int = 100, max_lag: int = 20
) -> np.ndarray:
    """
    Compute rolling Hurst exponent using R/S (Rescaled Range) analysis.
    
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending (persistent)
    """
    n = len(log_returns)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        segment = log_returns[i - window: i]
        if np.any(np.isnan(segment)) or np.std(segment) < 1e-12:
            continue

        try:
            lags = range(2, min(max_lag + 1, window // 2))
            tau = []
            for lag in lags:
                diffs = segment[lag:] - segment[:-lag]
                std_val = np.std(diffs)
                if std_val > 0:
                    tau.append(std_val)
                else:
                    tau.append(1e-12)

            if len(tau) >= 3:
                log_lags = np.log(list(lags[: len(tau)]))
                log_tau = np.log(tau)
                coeffs = np.polyfit(log_lags, log_tau, 1)
                result[i] = coeffs[0]  # Hurst exponent ≈ slope
        except (ValueError, np.linalg.LinAlgError):
            continue

    return result


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation of returns.
    """
    result = np.full_like(returns, np.nan, dtype=np.float64)
    for i in range(window, len(returns)):
        segment = returns[i - window: i]
        if not np.any(np.isnan(segment)):
            result[i] = np.std(segment)
    return result


def rolling_volatility_fast(returns: pd.Series, window: int) -> pd.Series:
    """
    Pandas-optimized rolling volatility for batch processing.
    """
    # Use population std to mirror numpy.std in the live pipeline.
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def vwap_deviation(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    """
    Price deviation from Volume-Weighted Average Price (VWAP).
    vwap_dev_t = (price_t - vwap_t) / vwap_t
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        p = prices[i - window: i]
        v = volumes[i - window: i]
        total_vol = np.sum(v)
        if total_vol > 0:
            vwap = np.sum(p * v) / total_vol
            if vwap > 0:
                result[i] = (prices[i] - vwap) / vwap

    return result


def vwap_deviation_fast(
    prices: pd.Series, volumes: pd.Series, window: int
) -> pd.Series:
    """
    Pandas-optimized VWAP deviation for batch processing.
    """
    pv = prices * volumes
    rolling_pv = pv.rolling(window=window, min_periods=window).sum()
    rolling_v = volumes.rolling(window=window, min_periods=window).sum()
    vwap = rolling_pv / rolling_v
    return (prices - vwap) / vwap


def trade_flow_imbalance(is_buyer_maker: np.ndarray, quantities: np.ndarray, window: int) -> np.ndarray:
    """
    Net buy vs sell aggressor ratio over a rolling window.
    Positive = net buying pressure, Negative = net selling pressure.
    
    In Binance data, is_buyer_maker=True means the trade was initiated by a seller
    (the buyer was the maker). So we invert: seller_volume when is_buyer_maker=True.
    """
    n = len(is_buyer_maker)
    result = np.full(n, np.nan, dtype=np.float64)

    buy_vol = np.where(~is_buyer_maker, quantities, 0.0)
    sell_vol = np.where(is_buyer_maker, quantities, 0.0)

    for i in range(window, n):
        bv = np.sum(buy_vol[i - window: i])
        sv = np.sum(sell_vol[i - window: i])
        total = bv + sv
        if total > 0:
            result[i] = (bv - sv) / total

    return result


def fractional_differentiation(series: np.ndarray, d: float = 0.5, threshold: float = 1e-5) -> np.ndarray:
    """
    Apply fractional differentiation of order `d` to a time series.
    Uses the expanding window method with weight truncation.
    
    This preserves more memory (long-range dependence) than integer differencing
    while still achieving stationarity.
    """
    n = len(series)
    
    # Compute weights
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k >= n:
            break
    
    weights = np.array(weights[::-1])  # Reverse for convolution
    width = len(weights)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(width - 1, n):
        segment = series[i - width + 1: i + 1]
        result[i] = np.dot(weights, segment)
    
    return result
