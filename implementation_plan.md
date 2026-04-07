# Polymarket BTC 5-Minute Market Trading Bot

## Goal

Build a gasless, ML-driven trading bot for Polymarket BTC 5-minute markets that:

- trains a LightGBM classifier on historical BTC data,
- computes the same feature schema in live trading,
- discovers the currently active BTC 5-minute market,
- submits post-only GTD maker orders only when the model has edge,
- enforces strict risk limits for a micro-capital bankroll.

This plan is based on the current repository snapshot from **2026-04-05**.

## Working Assumptions

- `LightGBM` is the only model stack for v1. No `MLX` dependency is needed.
- We start with a standard Polygon EOA and `SIGNATURE_TYPE=0` unless wallet requirements change.
- Historical training for v1 is kline-driven, with aggTrade ingestion kept as an enhancement path rather than a launch blocker.
- First production deployment targets Ubuntu 22.04+ with `systemd`.
- The bot should support a safe read-only validation mode before any live order placement.

## Current Status Snapshot

| Area | Status | Notes |
|---|---|---|
| Project scaffolding | Done | [`pyproject.toml`](pyproject.toml), [`requirements.txt`](requirements.txt), [`.gitignore`](.gitignore), and [`.env.example`](.env.example) are present. |
| Central configuration | Done | [`config/settings.py`](config/settings.py) defines Binance, Polymarket, feature, trading, risk, and path config. |
| Historical data fetch | Partial | [`scripts/01_fetch_historical.py`](scripts/01_fetch_historical.py) fetches klines and optional aggTrades with pagination. |
| Offline feature engineering | Partial | [`scripts/02_engineer_features.py`](scripts/02_engineer_features.py) builds a 20-feature training set from 1-minute kline proxies. |
| Model training | Partial | [`scripts/03_train_model.py`](scripts/03_train_model.py) trains LightGBM with `TimeSeriesSplit` and saves model metadata. |
| Model validation | Partial | [`scripts/04_validate_model.py`](scripts/04_validate_model.py) prints holdout metrics and calibration tables, but does not persist plots/artifacts. |
| Live feature pipeline | Partial | [`src/features/pipeline.py`](src/features/pipeline.py) computes the same named features, but the live methodology and buffer horizon need parity hardening. |
| Exchange connectivity | Partial | Binance WS, Gamma API, and Polymarket client wrappers exist in [`src/exchange/`](src/exchange/). |
| Execution engine | Partial | [`src/execution/engine.py`](src/execution/engine.py) wires ingestion, market discovery, inference, routing, and risk loops. |
| Risk controls | Partial | [`src/execution/risk_manager.py`](src/execution/risk_manager.py) has kill-switch, cooldown, position limit, and P&L floor logic, but accounting is not fully wired to fills yet. |
| Testing | Partial | `venv/bin/python -m pytest tests -q` passes locally (`61 passed` on 2026-04-05), but engine smoke tests and live adapter validation still need to run against real network responses. |
| Deployment | Partial | [`deploy/polymarket_bot.service`](deploy/polymarket_bot.service) and [`deploy/setup_server.sh`](deploy/setup_server.sh) are present, but still need first-run validation. |

## Critical Gaps Before Live Trading

### 1. Environment and dependency reproducibility

- The checked-in `venv` now runs on Python 3.12, and `venv/bin/python -m pytest tests -q` passes locally.
- We still need one documented, repeatable setup path on a fresh machine instead of relying on the checked-in environment.
- LightGBM and any native prerequisites should still be validated on a clean workstation and on the target Ubuntu host.

### 2. Feature parity between training and live inference

- Offline features are generated from 1-minute klines in [`scripts/02_engineer_features.py`](scripts/02_engineer_features.py).
- Live features in [`src/features/pipeline.py`](src/features/pipeline.py) are computed from trade ticks with approximated lookbacks.
- Feature names match, but the underlying windows and data sources are not yet guaranteed to be numerically comparable.

### 3. Rolling state horizon is too short for declared live windows

- [`src/utils/state.py`](src/utils/state.py) now reads `rolling_state_maxlen` from config and defaults to `50000`.
- Buffer capacity is no longer the immediate blocker, but the live feature pipeline still approximates minute windows from tick flow.
- We still need live-data validation that 30m and 60m features remain stable under real Binance trade density.

### 4. Trading safety and duplicate-order prevention

- The engine now has per-market duplicate-signal suppression and a dry-run path for safe local validation.
- Collateral balance/allowance checks are enforced before live order placement.
- Fill-derived P&L updates are still not wired into the trading loop, and read-only connectivity testing still needs an end-to-end smoke test.

### 5. Exchange response hardening

- Gamma and Polymarket wrapper code exists, but it still needs live schema validation against real responses.
- Order book parsing, market filtering, and credential bootstrap should be tested end to end before deployment.

## Phase Plan

### Phase 1: Baseline Environment and Verification

Objective: make the repo runnable, testable, and reproducible on one supported Python version.

Tasks:

- [ ] Recreate the local environment with Python 3.11 or 3.12.
- [ ] Install or document platform-native LightGBM prerequisites.
- [ ] Confirm `import lightgbm` works inside the project environment.
- [ ] Run the unit test suite successfully from the project environment.
- [ ] Document the exact setup commands in [`README.md`](README.md).

Acceptance criteria:

- `python -c "import lightgbm"` succeeds in the project venv.
- `python -m pytest tests -q` runs without import-time failures.
- The supported local Python version matches [`pyproject.toml`](pyproject.toml).

### Phase 2: Canonical Feature Contract

Objective: ensure training and live inference use the same feature schema, order, and intent.

Tasks:

- [ ] Define a single canonical feature list shared by training and inference.
- [ ] Decide whether v1 is strictly kline-based or whether aggTrade-derived features are required for launch.
- [ ] Align live lookback definitions with offline feature definitions.
- [ ] Increase rolling-state retention to cover the largest live window.
- [ ] Add tests that validate feature count, order, and numeric sanity across offline and live pipelines.

Acceptance criteria:

- Offline and live code use the same feature names in the same order.
- The live state buffer is large enough for the longest declared feature window.
- Feature-parity tests fail loudly if the schema drifts.

### Phase 3: Training Pipeline Hardening

Objective: make model training reproducible and auditable.

Tasks:

- [ ] Keep the current 20-feature schema unless feature parity work changes it intentionally.
- [ ] Persist training metadata with dataset date range, parameters, and feature names.
- [ ] Save validation summaries to disk instead of stdout only.
- [ ] Add a documented model promotion rule, such as minimum AUC/log-loss bounds or precision at a trading threshold.
- [ ] Decide whether to keep aggTrade collection optional or integrate it directly into training.

Acceptance criteria:

- A fresh model can be trained from raw data using documented commands only.
- The saved model loads successfully through [`src/execution/inference.py`](src/execution/inference.py).
- Training metadata is sufficient to reproduce the artifact later.

### Phase 4: Exchange Integration and Safe Order Flow

Objective: validate the full exchange path without taking unnecessary risk.

Tasks:

- [ ] Add a read-only mode that exercises Binance, Gamma, and Polymarket connectivity without placing orders.
- [ ] Verify live Gamma responses produce correct `yes_token_id` and `no_token_id` values.
- [ ] Validate Polymarket order book parsing against real response payloads.
- [ ] Add duplicate-signal suppression so the same market does not receive repeated orders every loop.
- [ ] Add balance and credential sanity checks before order placement.

Acceptance criteria:

- The engine can run end to end in read-only mode for at least 30 minutes without crashing.
- Market discovery yields a valid active BTC 5-minute market.
- The router places at most one intended order per market/signal condition.

### Phase 5: Risk and Accounting Completion

Objective: make sure the bot can stop itself safely when conditions deteriorate.

Tasks:

- [ ] Wire realized fills or resolved market outcomes into `RiskManager.update_pnl()`.
- [ ] Read max-position settings from config consistently everywhere.
- [ ] Block order placement when balances are insufficient.
- [ ] Log kill-switch activation, cooldown state, and trading halts in a way that is easy to inspect in production.
- [ ] Add tests for P&L floor breaches, cooldown recovery, and open-order caps under the final engine behavior.

Acceptance criteria:

- Trading stops immediately on P&L floor breach.
- Position limits and cooldowns are enforced by the engine, not only by unit tests.
- Risk state is visible in logs and periodic status reports.

### Phase 6: Deployment and Operations

Objective: make first production deployment boring and recoverable.

Tasks:

- [ ] Validate [`deploy/setup_server.sh`](deploy/setup_server.sh) on a clean Ubuntu host.
- [ ] Confirm [`deploy/polymarket_bot.service`](deploy/polymarket_bot.service) works with the final runtime paths and permissions.
- [ ] Add a first-run checklist for `.env`, model artifact copy, permissions, and service startup.
- [ ] Add an operator runbook for restart, log inspection, and safe shutdown.
- [ ] Decide whether production logs should remain plain JSON logging or move to full structured logging.

Acceptance criteria:

- A fresh server can bootstrap the app and start the service successfully.
- Restarting the service does not require manual code fixes.
- Operators have one clear checklist for setup and recovery.

## File-by-File Completion Targets

### Configuration and packaging

- [`pyproject.toml`](pyproject.toml): keep Python version, packaging metadata, and test settings aligned with the supported runtime.
- [`requirements.txt`](requirements.txt): confirm all packages are required and compatible with the supported Python version.
- [`.env.example`](.env.example): keep only the env vars that the final runtime actually reads.

### Data pipeline

- [`scripts/01_fetch_historical.py`](scripts/01_fetch_historical.py): confirm whether aggTrade fetching stays optional for v1.
- [`scripts/02_engineer_features.py`](scripts/02_engineer_features.py): treat this file as the source of truth for training features until a shared schema module exists.
- [`scripts/03_train_model.py`](scripts/03_train_model.py): preserve feature order exactly and persist all training metadata.
- [`scripts/04_validate_model.py`](scripts/04_validate_model.py): promote stdout-only validation into saved reports if the model is going to be retrained often.

### Live trading stack

- [`src/utils/state.py`](src/utils/state.py): enlarge buffer capacity and align time-based retrieval with actual feature horizon requirements.
- [`src/features/pipeline.py`](src/features/pipeline.py): enforce exact feature order and document any intentional approximations.
- [`src/execution/inference.py`](src/execution/inference.py): keep model load/predict path minimal and deterministic.
- [`src/execution/order_router.py`](src/execution/order_router.py): add duplicate-order suppression and any balance-aware routing guard.
- [`src/execution/risk_manager.py`](src/execution/risk_manager.py): complete P&L integration and config consistency.
- [`src/execution/engine.py`](src/execution/engine.py): add safe dry-run behavior and treat it as the main integration target.

### Exchange adapters

- [`src/exchange/binance_ws.py`](src/exchange/binance_ws.py): validate reconnection behavior and long-lived queue health.
- [`src/exchange/gamma_api.py`](src/exchange/gamma_api.py): verify real market filtering against current Polymarket naming patterns.
- [`src/exchange/polymarket_client.py`](src/exchange/polymarket_client.py): validate order placement, order-book access, and auth flows against real credentials.

### Deployment

- [`deploy/polymarket_bot.service`](deploy/polymarket_bot.service): confirm filesystem permissions and runtime paths on the target host.
- [`deploy/setup_server.sh`](deploy/setup_server.sh): verify that bootstrap steps match the final Python version and package setup.

## Verification Plan

### Automated checks

Run these after the environment is fixed:

```bash
python -m pytest tests/test_features.py -v
python -m pytest tests/test_inference.py -v
python -m pytest tests/test_order_router.py -v
python -m pytest tests/test_risk_manager.py -v
python -m pytest tests -q
```

### Data and model pipeline checks

```bash
python scripts/01_fetch_historical.py --days 30
python scripts/02_engineer_features.py
python scripts/03_train_model.py --n-splits 5
python scripts/04_validate_model.py
```

### Engine smoke tests

```bash
python -m src.execution.engine
```

Success conditions:

- no import errors,
- the model loads,
- Binance WS connects and fills the queue,
- Gamma discovery finds an active BTC market,
- the engine computes features without shape mismatch,
- risk status reports appear on schedule,
- no live orders are submitted during read-only validation.

## Definition of Done for v1

The bot is ready for a first controlled live trial when all of the following are true:

- [ ] Local environment setup is documented and reproducible.
- [x] Unit tests pass on the supported Python version.
- [x] Offline and live feature pipelines share one canonical schema.
- [x] A trained LightGBM model artifact exists and loads successfully.
- [ ] The engine runs end to end in read-only mode without crashes.
- [x] Balance checks, duplicate-order suppression, and risk halts are enforced.
- [ ] Deployment steps are validated on a clean Ubuntu server.
- [ ] Wallet credentials and Polymarket API credentials are confirmed working.

## Immediate Next Actions

1. Run the engine end to end in `DRY_RUN=true` mode with network access and confirm Binance/Gamma/Polymarket connectivity.
2. Validate Gamma market discovery, Polymarket order-book parsing, and balance checks against real credentialed responses.
3. Wire realized fills or resolved market outcomes into `RiskManager.update_pnl()`.
4. Save validation artifacts and define a model-promotion rule for retrains.
5. Validate the server bootstrap path on a clean Ubuntu machine.

## Current Verification Baseline

As of **2026-04-05**, `venv/bin/python -m pytest tests -q` completes successfully in this workspace under Python 3.12 with **61 passing tests**.

The workspace also already contains:

- `data/models/lgbm_btc_5m.txt`
- `data/models/training_metadata.json`
- `data/processed/features.parquet`

The main remaining blockers are now live network and credential validation, plus wiring realized P&L into the engine.
