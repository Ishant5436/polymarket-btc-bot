# Private Polymarket Research Agent (SolRouter Bounty)

This repository contains a **privacy-first Polymarket trading bot** specifically built for the **SolRouter bounty**.

## Why SolRouter?
In highly competitive 5-minute resolution markets like the Polymarket BTC "Price Up/Down" contracts, alpha decay happens in seconds. Using public LLM endpoints or public RPCs risks **front-running and strategy leaking**. 

By utilizing SolRouter and the private `gpt-oss-20b` models with encryption, this agent guarantees that our sentiment summaries, volume spike analysis, and liquidity sweep detections remain confidential, ensuring our strategy retains its edge without adversarial exploit.

## Quick Start
1. Add your SolRouter API key to the `.env` file:
   ```bash
   SOLROUTER_API_KEY=your_secure_key_here
   ```
2. Run the secure research agent:
   ```bash
   node agent.js
   ```

---

*The contents below represent the pre-existing, underlying Python pipeline for LightGBM inference and execution on the Polygon network.*

---
# Polymarket BTC 5-Minute Market Trading Bot

ML-driven, gasless trading bot that exploits mispricings in Polymarket's 5-minute BTC resolution markets using LightGBM inference, real-time Binance price streams, and post-only maker execution on Polygon.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Binance WS     │────▶│  Rolling State    │────▶│  Feature        │
│  btcusdt@agg    │     │  (deque, 50k)     │     │  Pipeline       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
┌─────────────────┐                                       ▼
│  Gamma API      │     ┌──────────────────┐     ┌─────────────────┐
│  Market Finder  │────▶│  LightGBM        │◀────│  Feature Vector │
└─────────────────┘     │  Inference       │     │  (20 features)  │
                        └────────┬─────────┘     └─────────────────┘
                                 │ p̂ᵢ
                                 ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Order Router    │────▶│  Polymarket     │
                        │  Edge Detection  │     │  CLOB (post_only│
                        └──────────────────┘     │  GTD limit)     │
                                                 └─────────────────┘
                        ┌──────────────────┐
                        │  Risk Manager    │──── Kill-Switch / Cancel
                        │  σ Monitor       │
                        └──────────────────┘
```

## Quick Start

### 1. Environment Setup

#### MacOS (Recommended)
```bash
# Install Homebrew, Python 3.12, and libomp (required for LightGBM)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 libomp

# Setup project
cd /path/to/BTC
rm -rf venv/ && python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Linux/Other
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Full Data Pipeline (One Command)
Run the entire fetch-engineer-train-validate sequence in one go:

```bash
python scripts/01_fetch_historical.py --days 90 && \
python scripts/02_engineer_features.py && \
python scripts/03_train_model.py --n-splits 5 && \
python scripts/04_validate_model.py
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Start the Bot (Local)

```bash
# Recommended for first connectivity validation
export DRY_RUN=true

python -m src.execution.engine
```

### 4a. Start With a $1 Budget

If you want the bot to target a strict `$1` spend per order, use the
dollar-based sizing path instead of raw shares:

```bash
export DRY_RUN=false
export LIVE_TRADING_ENABLED=true
export ORDER_NOTIONAL=1.0
export MAX_ORDER_NOTIONAL=1.0
export BANKROLL_FRACTION_PER_ORDER=1.0
export ALLOW_UPSIZE_TO_MIN_ORDER_SIZE=false
```

Polymarket venue minimums still apply. If a live market needs more than `$1`
to satisfy its minimum order size, the bot will skip that trade instead of
upsizing past your cap.

### 4b. Run Validation Preflight

Before enabling live orders, run the account-aware preflight:

```bash
venv/bin/python scripts/05_validation_preflight.py --budget 1.0
```

Then run the full engine in authenticated read-only mode:

```bash
venv/bin/python -m src.execution.engine --validation-only
```

### 5. Deploy to Server

```bash
# On server (Ubuntu 22.04+)
sudo ./deploy/setup_server.sh <git-repo-url>

# Copy model and .env, then:
sudo systemctl start polymarket_bot
journalctl -u polymarket_bot -f
```

## Features

| Feature | Description |
|---------|-------------|
| **Order Book Imbalance** | Buy/sell volume ratio from tick data |
| **Micro-Price Momentum** | Log returns over 1m, 2m, 5m, 10m windows |
| **Hurst Exponent** | Rolling R/S analysis for trend detection |
| **Fractional Differentiation** | Memory-preserving stationarization |
| **Rolling Volatility** | σ across multiple timeframes |
| **VWAP Deviation** | Price distance from volume-weighted average |
| **Trade Flow Imbalance** | Net buy/sell aggressor pressure |

## Risk Controls

- **Volatility Kill-Switch**: Cancels all orders on 3σ vol breach
- **P&L Floor**: Halts trading at -$0.20 cumulative loss
- **Position Limits**: Max 3 concurrent resting orders
- **Post-Only Enforcement**: All orders are maker-only (zero fees)
- **GTD Expiration**: Orders expire after 10 seconds

## Fee Model

| Action | Fee |
|--------|-----|
| Maker (post_only) | **0%** + eligible for rebates |
| Taker | ~1.56% (dynamic) |
| Gas (Polygon) | **$0** (covered by Relayer) |

https://github.com/user-attachments/assets/27dd02db-e4be-4ed3-9eda-855a5db22f2c

## License

MIT
