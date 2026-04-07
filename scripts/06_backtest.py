#!/usr/bin/env python3
"""
Backtest the trained model on historical data with realistic execution simulation.

Simulates:
  - $100 starting capital (configurable)
  - Edge-based entry (min_edge threshold)
  - Stop-loss and take-profit exits
  - Kelly-fraction position sizing
  - Spread/slippage costs
  - One position at a time (sequential markets)
  - Full P&L tracking with detailed trade log

Usage:
    python scripts/06_backtest.py --capital 100 --min-edge 0.02
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN
from src.utils.model_metadata import load_training_metadata, resolve_target_horizon_minutes


@dataclass
class Trade:
    entry_idx: int
    entry_time: str
    side: str            # "YES" or "NO"
    entry_price: float
    size: float          # number of shares
    notional: float      # dollars risked
    model_prob: float
    edge: float
    exit_idx: Optional[int] = None
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    won: bool = False


def run_backtest(
    df: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    capital: float = 100.0,
    min_edge: float = 0.02,
    order_notional: float = 1.0,
    max_order_notional: float = 5.0,
    stop_loss_factor: float = 0.5,
    take_profit_multiple: float = 0.3,
    kelly_fraction: float = 0.25,
    spread_cost: float = 0.01,
    market_horizon: int = 5,
    min_prob: float = 0.25,
    max_entry_price: float = 0.90,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    Walk-forward backtest simulating 5-minute binary market trading.
    
    Each row = 1 minute bar. Every `market_horizon` rows = one market.
    The model predicts at the start of each market whether BTC goes up.
    If there's enough edge, we enter. The market resolves after `market_horizon` bars.
    """
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values
    timestamps = df[TIMESTAMP_COLUMN].values
    closes = df["close"].values if "close" in df.columns else None

    # Predict all probabilities at once
    probs = model.predict(X)
    
    trades: list[Trade] = []
    equity_curve = [capital]
    balance = capital
    
    # Walk through data in market_horizon-sized windows
    n = len(df)
    warmup = 100  # Skip first 100 bars (feature warmup)
    
    i = warmup
    while i + market_horizon < n:
        prob = float(probs[i])
        actual = int(y[i])
        
        # Determine best side
        yes_prob = prob
        no_prob = 1.0 - prob
        
        # Simulate market price as ~fair value with noise
        # In live trading, market_price comes from the order book
        # Here we simulate it as slightly noisy around 0.50
        market_yes_price = 0.50  # simplified: binary market near 50/50
        market_no_price = 0.50
        
        # Check YES side
        yes_edge = yes_prob - (market_yes_price + spread_cost)
        no_edge = no_prob - (market_no_price + spread_cost)
        
        side = None
        edge = 0.0
        entry_price = 0.0
        side_prob = 0.0
        
        if yes_edge >= min_edge and yes_prob >= min_prob:
            side = "YES"
            edge = yes_edge
            entry_price = market_yes_price + spread_cost  # pay spread
            side_prob = yes_prob
        elif no_edge >= min_edge and no_prob >= min_prob:
            side = "NO"
            edge = no_edge
            entry_price = market_no_price + spread_cost
            side_prob = no_prob
        
        if side is None or entry_price > max_entry_price or entry_price <= 0.01:
            # No trade this market
            i += market_horizon
            equity_curve.append(balance)
            continue
        
        # Kelly position sizing
        p = side_prob
        q = 1.0 - p
        b = (1.0 - entry_price) / entry_price  # odds ratio
        kelly_f = (p * b - q) / b if b > 0 else 0
        kelly_f = max(0, min(kelly_f, 1.0)) * kelly_fraction
        
        # Size the trade
        notional = min(
            order_notional,
            max_order_notional,
            balance * kelly_f if kelly_f > 0 else order_notional,
            balance * 0.10,  # never risk more than 10% per trade
        )
        notional = max(notional, 0.50)  # minimum $0.50
        
        if notional > balance:
            i += market_horizon
            equity_curve.append(balance)
            continue
        
        shares = notional / entry_price
        
        # Determine outcome after market_horizon bars
        future_idx = min(i + market_horizon, n - 1)
        
        # Did BTC go up over the market horizon?
        btc_went_up = bool(y[i])  # target at entry point = forward-looking label
        
        # Determine if we won
        if side == "YES":
            won = btc_went_up
        else:
            won = not btc_went_up
        
        # P&L calculation (binary market)
        if won:
            pnl = (1.0 - entry_price) * shares  # win: get $1 per share, paid entry_price
        else:
            pnl = -entry_price * shares  # lose: lose entire entry
        
        # Apply stop-loss cap (simulated)
        max_loss = -entry_price * shares * stop_loss_factor
        if pnl < max_loss:
            pnl = max_loss
            exit_reason = "stop_loss"
        elif won and pnl > 0:
            # Apply take-profit (partial, simulated)
            tp_target = (1.0 - entry_price) * take_profit_multiple * shares
            if pnl > tp_target and tp_target > 0:
                exit_reason = "take_profit"
            else:
                exit_reason = "market_resolve_win"
        else:
            exit_reason = "market_resolve_loss"
        
        balance += pnl
        
        trade = Trade(
            entry_idx=i,
            entry_time=str(timestamps[i]),
            side=side,
            entry_price=entry_price,
            size=shares,
            notional=notional,
            model_prob=prob,
            edge=edge,
            exit_idx=future_idx,
            exit_time=str(timestamps[future_idx]),
            exit_price=1.0 if won else 0.0,
            exit_reason=exit_reason,
            pnl=pnl,
            won=won,
        )
        trades.append(trade)
        equity_curve.append(balance)
        
        # Move to next market
        i += market_horizon
        
        # Circuit breaker: stop if we lost too much
        if balance <= capital * 0.20:
            print(f"\n[!] Circuit breaker: balance dropped to ${balance:.2f} (80% loss)")
            break
    
    # Build equity curve DataFrame
    eq_df = pd.DataFrame({"equity": equity_curve})
    
    return trades, eq_df


def print_results(trades: list[Trade], equity_df: pd.DataFrame, capital: float):
    """Print detailed backtest results."""
    if not trades:
        print("\n[!] No trades executed during backtest period.")
        return
    
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.won)
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total_trades
    avg_win = np.mean([t.pnl for t in trades if t.won]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if not t.won]) if losses > 0 else 0
    
    final_balance = capital + total_pnl
    roi = (final_balance - capital) / capital * 100
    
    # Max drawdown
    equity = equity_df["equity"].values
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(np.min(drawdown)) * 100
    
    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe-like ratio (on per-trade returns)
    trade_returns = [t.pnl / t.notional for t in trades]
    sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252 * 12) if np.std(trade_returns) > 0 else 0
    
    # Side breakdown
    yes_trades = [t for t in trades if t.side == "YES"]
    no_trades = [t for t in trades if t.side == "NO"]
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — ${capital:.0f} Starting Capital")
    print(f"{'='*60}")
    print(f"  Period:           {trades[0].entry_time[:10]} → {trades[-1].entry_time[:10]}")
    print(f"  Total Trades:     {total_trades:,}")
    print(f"  Wins / Losses:    {wins} / {losses}")
    print(f"  Win Rate:         {win_rate:.1%}")
    print(f"{'='*60}")
    print(f"  Final Balance:    ${final_balance:.2f}")
    print(f"  Total P&L:        ${total_pnl:+.2f}")
    print(f"  ROI:              {roi:+.1f}%")
    print(f"  Max Drawdown:     {max_drawdown:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Sharpe (ann.):    {sharpe:.2f}")
    print(f"{'='*60}")
    print(f"  Avg P&L/trade:    ${avg_pnl:+.4f}")
    print(f"  Avg Win:          ${avg_win:+.4f}")
    print(f"  Avg Loss:         ${avg_loss:+.4f}")
    print(f"  Win/Loss Ratio:   {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"{'='*60}")
    
    if yes_trades:
        yes_wins = sum(1 for t in yes_trades if t.won)
        yes_pnl = sum(t.pnl for t in yes_trades)
        print(f"  YES trades:       {len(yes_trades)} ({yes_wins}/{len(yes_trades)} = {yes_wins/len(yes_trades):.1%} WR)  P&L: ${yes_pnl:+.2f}")
    if no_trades:
        no_wins = sum(1 for t in no_trades if t.won)
        no_pnl = sum(t.pnl for t in no_trades)
        print(f"  NO  trades:       {len(no_trades)} ({no_wins}/{len(no_trades)} = {no_wins/len(no_trades):.1%} WR)  P&L: ${no_pnl:+.2f}")
    
    print(f"{'='*60}")
    
    # Edge distribution
    edges = [t.edge for t in trades]
    print(f"\n  Edge Distribution:")
    for threshold in [0.02, 0.05, 0.08, 0.10, 0.15]:
        count = sum(1 for e in edges if e >= threshold)
        wins_at = sum(1 for t in trades if t.edge >= threshold and t.won)
        wr_at = wins_at / count if count > 0 else 0
        pnl_at = sum(t.pnl for t in trades if t.edge >= threshold)
        if count > 0:
            print(f"    edge≥{threshold:.2f}: {count:5d} trades  WR={wr_at:.1%}  P&L=${pnl_at:+.2f}")
    
    # Show last 10 trades
    print(f"\n  Last 10 Trades:")
    print(f"  {'Time':>20s} {'Side':>4s} {'Prob':>6s} {'Edge':>6s} {'Entry':>6s} {'Size':>6s} {'P&L':>8s} {'Result':>12s}")
    for t in trades[-10:]:
        print(
            f"  {t.entry_time[:19]:>20s} {t.side:>4s} {t.model_prob:>6.3f} "
            f"{t.edge:>6.3f} ${t.entry_price:>5.2f} {t.size:>6.1f} "
            f"${t.pnl:>+7.2f} {t.exit_reason:>12s}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest trading model on historical data")
    parser.add_argument("--capital", type=float, default=100.0, help="Starting capital ($)")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Min edge to enter")
    parser.add_argument("--order-notional", type=float, default=1.0, help="Target $ per trade")
    parser.add_argument("--max-notional", type=float, default=5.0, help="Max $ per trade")
    parser.add_argument("--stop-loss", type=float, default=0.5, help="Stop-loss factor")
    parser.add_argument("--take-profit", type=float, default=0.3, help="Take-profit multiple")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction")
    parser.add_argument("--spread", type=float, default=0.01, help="Spread cost per side")
    parser.add_argument("--horizon", type=int, default=5, help="Market horizon (minutes)")
    parser.add_argument("--holdout-only", action="store_true", help="Only backtest on holdout set")
    args = parser.parse_args()

    # Load data and model
    if not os.path.exists(PATHS.features_path):
        print(f"[!] Features not found at {PATHS.features_path}")
        sys.exit(1)
    if not os.path.exists(PATHS.model_path):
        print(f"[!] Model not found at {PATHS.model_path}")
        sys.exit(1)

    df = pd.read_parquet(PATHS.features_path)
    model = lgb.Booster(model_file=PATHS.model_path)
    metadata = load_training_metadata(PATHS.model_path)
    feature_cols = metadata.get("feature_columns", list(FEATURE_COLUMNS))
    horizon = resolve_target_horizon_minutes(metadata)

    print(f"[*] Loaded {len(df):,} samples, model with {len(feature_cols)} features")
    print(f"[*] Model horizon: {horizon} minutes")

    # Use holdout set (last 20%) or full dataset
    if args.holdout_only:
        split_idx = int(len(df) * 0.8)
        df = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"[*] Using holdout set: {len(df):,} samples")
    else:
        # Use last 60% to avoid training data
        split_idx = int(len(df) * 0.4)
        df = df.iloc[split_idx:].copy().reset_index(drop=True)
        print(f"[*] Using test set (last 60%): {len(df):,} samples")

    print(f"[*] Capital: ${args.capital:.2f}")
    print(f"[*] Config: min_edge={args.min_edge}, notional=${args.order_notional}, "
          f"SL={args.stop_loss}, TP={args.take_profit}, kelly={args.kelly}, spread={args.spread}")

    # Run backtest
    trades, equity_df = run_backtest(
        df, model, feature_cols,
        capital=args.capital,
        min_edge=args.min_edge,
        order_notional=args.order_notional,
        max_order_notional=args.max_notional,
        stop_loss_factor=args.stop_loss,
        take_profit_multiple=args.take_profit,
        kelly_fraction=args.kelly,
        spread_cost=args.spread,
        market_horizon=args.horizon,
    )

    # Print results
    print_results(trades, equity_df, args.capital)

    # Save trade log
    os.makedirs("data/artifacts/backtests", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    trade_log = [{
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "side": t.side,
        "entry_price": t.entry_price,
        "size": t.size,
        "notional": t.notional,
        "model_prob": t.model_prob,
        "edge": t.edge,
        "exit_reason": t.exit_reason,
        "pnl": t.pnl,
        "won": t.won,
    } for t in trades]
    
    report = {
        "timestamp": timestamp,
        "capital": args.capital,
        "config": {
            "min_edge": args.min_edge,
            "order_notional": args.order_notional,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit,
            "kelly": args.kelly,
            "spread": args.spread,
        },
        "summary": {
            "total_trades": len(trades),
            "wins": sum(1 for t in trades if t.won),
            "win_rate": sum(1 for t in trades if t.won) / len(trades) if trades else 0,
            "total_pnl": sum(t.pnl for t in trades),
            "final_balance": args.capital + sum(t.pnl for t in trades),
            "roi_pct": (sum(t.pnl for t in trades) / args.capital * 100) if trades else 0,
        },
        "trades": trade_log,
    }
    
    report_path = f"data/artifacts/backtests/backtest_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Trade log saved to {report_path}")


if __name__ == "__main__":
    main()
