# Pre-Live Checklist

Use this sequence before enabling real orders.

## 1. Account Preflight

Run the wallet-aware preflight first:

```bash
venv/bin/python scripts/05_validation_preflight.py --budget 1.0
```

Go/no-go:

- `Status: READY_FOR_VALIDATION_ONLY`
- model loads successfully
- collateral balance and allowance are readable
- an active BTC market is discovered
- you understand whether the current venue minimum fits inside your budget

## 2. Validation-Only Run

Run the full engine with authenticated checks but no order placement:

```bash
venv/bin/python -m src.execution.engine --validation-only
```

Watch for:

- Binance connects and streams trades
- Gamma finds the active BTC market
- Polymarket order book requests succeed
- no crashes for at least 30-60 minutes
- no unexpected open orders appear in your account

## 3. Live Trial Gate

Only consider live mode if all of the following are true:

- the preflight passes cleanly
- the validation-only run stays stable for at least 30-60 minutes
- the current market minimum notional is compatible with your intended budget
- you have confirmed your wallet starts with the collateral you are willing to risk
- you are actively supervising the first live session

## 4. First Live Trial

Before the first live trial:

- set `DRY_RUN=false`
- set `VALIDATION_ONLY_MODE=false`
- set `LIVE_TRADING_ENABLED=true`
- keep your `ORDER_NOTIONAL` and `MAX_ORDER_NOTIONAL` cap intentional

Suggested first-live posture:

- keep the account isolated from unrelated manual trading
- start with a tiny bankroll
- supervise logs continuously
- stop immediately if you see repeated rejections, unexpected open orders, or unexplained P&L changes
