# Gatekeeping Logs Guide

## Overview
The bot now logs detailed "gatekeeping" messages that show exactly **why** each potential trade is being rejected or accepted. These logs use `[SKIP_*]` and `[SIGNAL]` prefixes to make filtering patterns immediately visible.

## Log Format

### When a Trade is Rejected
```
[SKIP_YES] Insufficient edge | model_p=0.5400 yes_ask=0.5300 edge=0.0100 min_edge=0.0200 | need +0.0100 more
```

Key fields:
- **[SKIP_YES/NO]**: Rejected signal (YES or NO side)
- **model_p**: Model's predicted probability
- **edge**: Actual edge (model_prob - market_price)
- **min_edge**: Minimum edge required (your 2% threshold)
- **need +X more**: How much more edge is required

### When a Trade is Accepted
```
[SIGNAL] YES side accepted | model_p=0.5600 yes_ask=0.5300 edge=0.0300 min_edge=0.0200 price=0.53 size=10.25
```

## Filter Types

### 1. **[SKIP_*] Insufficient edge**
**Why it happens**: The bot's edge is below the 2% threshold.
```
[SKIP_YES] Insufficient edge | ... edge=0.0080 min_edge=0.0200 | need +0.0120 more
```
**How to fix**: Lower `min_edge` in config (e.g., 2% → 1%)

### 2. **[SKIP_*] Model prob too low**
**Why it happens**: Prediction is too close to 50/50 (lacks conviction).
```
[SKIP_YES] Model prob too low | model_p=0.5150 min_side_prob=0.5200 | need +0.0050 higher
```
**How to fix**: Lower `min_side_probability` in config (e.g., 52% → 51%)

### 3. **[SKIP_*] Order book imbalance too low**
**Why it happens**: Ask-side liquidity is too heavy; market looks bearish for a BUY.
```
[SKIP_YES] Order book imbalance too low | imbalance=0.3200 min=0.3500 | need +0.0300 higher
```
**How to fix**: Lower `min_order_book_imbalance` in config (e.g., 35% → 30%)

### 4. **[SKIP_*] Ask wall resistance too high**
**Why it happens**: There's a huge ask wall that would block the order fill.
```
[SKIP_YES] Ask wall resistance too high | ask_wall_ratio=2.8 max=2.5
```
**How to fix**: Increase `max_ask_wall_ratio` in config (e.g., 2.5 → 3.0)

### 5. **[SKIP_*] Entry price too high**
**Why it happens**: The calculated limit price is too expensive.
```
[SKIP_YES] Entry price too high | price=0.9100 max_entry_price=0.9000 | need 0.0100 price reduction
```
**How to fix**: Increase `max_entry_price` in config (e.g., 90% → 92%)

### 6. **[SKIP] Not enough time remaining**
**Why it happens**: Market is expiring soon; not enough time to exit profitably.
```
[SKIP] Not enough time remaining | market=btc-5m-apr-8 min_seconds=60
```
**How to fix**: Lower `min_time_remaining_seconds` in config (e.g., 60 → 30)

### 7. **[KILL_SWITCH] VOLATILITY KILL-SWITCH TRIGGERED**
**Why it happens**: Bitcoin made a sudden large move; bot froze trading for 60 seconds.
```
[KILL_SWITCH] VOLATILITY KILL-SWITCH TRIGGERED | current_vol=0.00250 baseline_vol=0.00100 
rel_multiplier=2.50x z_score=3.50 threshold=2.5σ | trading paused for 60 seconds
```
**How to fix**: Adjust volatility thresholds in `config/settings.py` under `RiskConfig`

## Interpreting Filter Patterns

### Pattern 1: Dominated by "Insufficient edge"
```
Filter rejection counts:
  edge_too_low: 847 (74.3%)
  prob_too_low: 120 (10.5%)
  imbalance_too_low: 58 (5.1%)
  ...
```
**Diagnosis**: The market is efficiently priced. Your 2% edge threshold is too strict.  
**Action**: Lower `min_edge` to 1% and re-test.

### Pattern 2: Dominated by "Model prob too low"
```
Filter rejection counts:
  prob_too_low: 312 (68.9%)
  edge_too_low: 89 (19.6%)
  ...
```
**Diagnosis**: Your model predictions cluster around 50% (0–2% conviction).  
**Action**: Review model training; increase `min_side_probability` threshold incrementally (e.g., 52% → 51%).

### Pattern 3: Dominated by "Imbalance too low"
```
Filter rejection counts:
  imbalance_too_low: 245 (52.3%)
  edge_too_low: 198 (42.4%)
  ...
```
**Diagnosis**: Most trading opportunities occur during ask-heavy periods (market weakness).  
**Action**: Lower `min_order_book_imbalance` to 30% or 25%.

### Pattern 4: High "Kill-switch" events
```
[KILL_SWITCH] VOLATILITY KILL-SWITCH TRIGGERED (14 times in 90 minutes)
```
**Diagnosis**: Bitcoin is in a trending market; volatility spikes are freezing the bot.  
**Action**: Adjust `volatility_sigma_threshold` or `volatility_min_absolute_threshold` in RiskConfig.

## How to Use These Logs

1. **Run your bot for 30–60 minutes** with active Polymarket activity.
2. **Grep for filter patterns**:
   ```bash
   tail -f bot.log | grep "\[SKIP"
   ```
3. **At the end, check the summary**:
   ```bash
   tail -f bot.log | grep "GATEKEEPING FILTER SUMMARY" -A 20
   ```
4. **Identify the top 2–3 most restrictive filters**.
5. **Make a targeted adjustment**: Lower the threshold just slightly for the #1 filter.
6. **Re-run and check if trade frequency improves**.

## Example Workflow

**Initial Run (1 hour)**:
- 47 trade opportunities evaluated
- Result: 0 orders placed
- Filter summary:
  ```
  edge_too_low: 32 (68.1%)
  prob_too_low: 10 (21.3%)
  imbalance_too_low: 5 (10.6%)
  ```

**Decision**: "Edge is crushing us. Reduce from 2% to 1%."

**After adjustment** (run again):
- 52 trade opportunities
- Result: 8 orders placed ✓
- Filter summary:
  ```
  edge_too_low: 12 (24%)
  prob_too_low: 28 (56%)
  imbalance_too_low: 10 (20%)
  ```

**Next decision**: "Probability gate is now #1. Let's check model predictions..."

## Key Config Parameters to Adjust

| Parameter | File | Default | Impact |
|-----------|------|---------|--------|
| `min_edge` | `settings.py` | 0.02 | Minimum edge % required |
| `min_side_probability` | `settings.py` | 0.52 | Min conviction (>50%) |
| `min_order_book_imbalance` | `settings.py` | 0.35 | Min bid-side weight |
| `max_ask_wall_ratio` | `settings.py` | 2.5 | Max ask/bid size ratio |
| `max_entry_price` | `settings.py` | 0.90 | Max entry price as % of 1.0 |
| `min_time_remaining_seconds` | `settings.py` | 60 | Min time until expiry |
| `volatility_sigma_threshold` | `settings.py` | 2.5 | Kill-switch Z-score |

## Next Steps

After implementing these gatekeeping logs:

1. **Run the bot for 1–2 hours** to generate enough data.
2. **Review the log output** and identify the top filtering constraints.
3. **Recommend a single-parameter adjustment** based on the data.
4. **Test and measure the impact** on order frequency.
5. **Iterate until hitting your target trade frequency** (e.g., 1–2 orders per hour).
