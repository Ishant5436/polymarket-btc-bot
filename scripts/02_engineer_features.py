#!/usr/bin/env python3
"""
02_engineer_features.py
Reads raw historical BTC data and computes the full feature set for model training.
Outputs a Parquet file with features and binary target labels.

Usage:
    python scripts/02_engineer_features.py [--target-horizon-minutes 5]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import FEATURES, PATHS
from src.features.minute_features import compute_feature_frame
from src.features.schema import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN


def load_klines() -> pd.DataFrame:
    """Load raw kline CSV and validate columns."""
    path = PATHS.klines_path
    if not os.path.exists(path):
        print(f"[!] Kline file not found at {path}")
        print("    Run scripts/01_fetch_historical.py first.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    print(f"[*] Loaded {len(df):,} klines from {path}")
    print(f"    Date range: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from kline data and return enriched DataFrame.
    """
    print("[*] Computing features...")
    df = compute_feature_frame(df)
    print("    [✓] Feature frame computed from shared minute-bar logic")
    return df


def compute_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Binary target: does BTC price increase over the next `horizon` minutes?
    1 = price went up, 0 = price went down or stayed flat
    """
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    print(f"[*] Target: {horizon}-minute forward return > 0")
    print(f"    Class distribution: {df['target'].value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Engineer model features for BTC prediction")
    parser.add_argument(
        "--target-horizon-minutes",
        type=int,
        default=5,
        help="Forward return horizon used to build the binary target",
    )
    args = parser.parse_args()

    # Load raw data
    df = load_klines()

    # Compute features
    df = compute_features(df)

    # Compute target
    df = compute_target(df, horizon=args.target_horizon_minutes)

    # Define feature columns
    feature_cols = list(FEATURE_COLUMNS)

    # Keep only feature columns + target + timestamp
    output_cols = [TIMESTAMP_COLUMN] + feature_cols + [TARGET_COLUMN, "future_return"]
    df_out = df[output_cols].copy()

    # Drop rows with NaN (from rolling windows)
    n_before = len(df_out)
    df_out.dropna(subset=feature_cols + [TARGET_COLUMN], inplace=True)
    n_after = len(df_out)
    print(f"\n[*] Dropped {n_before - n_after:,} rows with NaN ({n_before:,} → {n_after:,})")

    # Save
    os.makedirs(PATHS.processed_data_dir, exist_ok=True)
    df_out.to_parquet(PATHS.features_path, index=False)
    print(f"[✓] Saved features to {PATHS.features_path}")
    print(f"    Shape: {df_out.shape}")
    print(f"    Features: {feature_cols}")
    print(
        f"    Target balance: "
        f"{df_out[TARGET_COLUMN].value_counts(normalize=True).to_dict()}"
    )
    print(f"    Target horizon minutes: {args.target_horizon_minutes}")


if __name__ == "__main__":
    main()
