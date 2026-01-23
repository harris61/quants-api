# Trading Strategy Documentation

## Overview

**Strategy Name:** MA50 + Momentum + Foreign Flow Daily Ranking
**Model Version:** `rule_ma50_v2`
**Target:** Indonesian Stock Exchange (IDX) equities
**Objective:** Identify stocks likely to gain ≥5% in the next trading day

---

## Strategy Logic

### Core Concept

The strategy combines **trend-following** (MA50) with **momentum** and **foreign flow** to rank stocks daily. It filters for stocks in uptrends, applies a movers filter for active stocks, and scores them by momentum, trend strength, volume confirmation, and foreign investor activity.

### Pre-Filter: Movers Filter

Before scoring, stocks must appear in daily movers lists:
- `top_value` - Highest trading value
- `top_volume` - Highest trading volume
- `top_frequency` - Most transactions

This ensures we only trade actively traded stocks. Controlled by `MOVERS_FILTER_ENABLED = True`.

### Entry Criteria (Hard Filters)

A stock must pass ALL filters to be considered:

| Filter | Condition | Config Parameter | Purpose |
|--------|-----------|------------------|---------|
| Above MA50 | `close > MA50` | `RULE_DIST50_MIN = 0.0` | Only long in uptrends |
| Not Overextended | `dist50 < 15%` | `RULE_DIST50_MAX = 0.15` | Avoid chasing |
| Slope Not Falling | `slope50 > -0.2%` | `RULE_SLOPE_FLAT_MIN = -0.002` | MA50 not declining |

Where:
- `dist50 = (close - MA50) / MA50` — percentage distance from MA50
- `slope50 = (MA50 - MA50[5 days ago]) / MA50[5 days ago]` — 5-day momentum of MA50

### Scoring System

Stocks passing the filters are scored (0-1 scale) using five components:

| Component | Weight | Range | Config Parameters |
|-----------|--------|-------|-------------------|
| **Momentum** | 32% | 5-day return: -5% to +10% | `RULE_MOMENTUM_FLOOR`, `RULE_MOMENTUM_CEIL` |
| **Slope** | 23% | MA50 slope: -0.1% to +0.5% | `RULE_SLOPE_SCORE_FLOOR`, `RULE_SLOPE_SCORE_CEIL` |
| **Dist50** | 18% | Distance above MA50: 0% to 10% | `RULE_SCORE_DIST50_CAP` |
| **Volume** | 17% | Volume ratio: 0.5x to 1.5x avg | `RULE_VOLUME_RATIO_FLOOR`, `RULE_VOLUME_RATIO_CEIL` |
| **Foreign Flow** | 10% | Net foreign: -1B to +10B IDR | `RULE_FOREIGN_FLOOR`, `RULE_FOREIGN_CEIL` |

**Score Formula:**
```
foreign_weight = 10 if foreign_net is available else 0
score = (
    momentum_score * 32
    + slope_score * 23
    + dist50_score * 18
    + volume_score * 17
    + foreign_score * foreign_weight
) / (32 + 23 + 18 + 17 + foreign_weight)
```


Each component is clamped to [0, 1]:
```python
momentum_score = clamp((momentum - (-0.05)) / (0.10 - (-0.05)))  # -5%→0, +10%→1
slope_score    = clamp((slope50 - (-0.001)) / (0.005 - (-0.001)))
dist50_score   = clamp(dist50 / 0.10)                            # 0%→0, 10%→1
volume_score   = clamp((volume_ratio - 0.5) / (1.5 - 0.5))       # 0.5x→0, 1.5x→1
foreign_score  = clamp((foreign_net - (-1B)) / (10B - (-1B)))    # -1B→0, +10B→1
```

**Note:** When foreign flow data is unavailable for a stock, its foreign weight is set to 0 (no contribution).

### Daily Output

The strategy outputs the **top 5 stocks** ranked by score (max 5), with:
- Symbol and name
- Score (0-1)
- 5-day momentum (%)
- Volume ratio (x avg)
- Foreign net flow (if available)

### Corporate Actions Adjustments

If corporate actions are present in the database, the system applies backward adjustments:
- **Stock splits**: Prices before ex-date are divided by split ratio; volumes are multiplied.
- **Cash dividends**: Prices before ex-date are reduced by the cash dividend amount (floored).

These adjustments are applied before indicators are computed to avoid artificial gaps.

---

## Configuration Reference

All parameters are in `config.py`:

```python
# ==================== STRATEGY SETTINGS ====================
TOP_GAINER_THRESHOLD = 0.05          # 5% target for "correct" prediction
TOP_PICKS_COUNT = 5                  # Top N stocks per day

# ==================== MOVERS FILTER ====================
MOVERS_FILTER_ENABLED = True         # Filter to active stocks only
MOVERS_FILTER_TYPES = ["top_value", "top_volume", "top_frequency"]

# ==================== RULE-BASED SETTINGS ====================
RULE_BASED_MODEL_NAME = "rule_ma50_v2"
RULE_MA_SLOW = 50                    # MA period
RULE_SLOPE_LOOKBACK = 5              # Days for slope calculation

# Filter thresholds
RULE_DIST50_MIN = 0.0                # Must be above MA50
RULE_DIST50_MAX = 0.15               # Max 15% above MA50
RULE_SLOPE_FLAT_MIN = -0.002         # MA50 slope floor (-0.2%)

# Momentum settings
RULE_MOMENTUM_PERIOD = 5             # 5-day return
RULE_MOMENTUM_FLOOR = -0.05          # -5% scores 0
RULE_MOMENTUM_CEIL = 0.10            # +10% scores 1

# Volume settings
RULE_VOLUME_MA_PERIOD = 20           # Volume MA period
RULE_VOLUME_RATIO_FLOOR = 0.5        # 0.5x avg scores 0
RULE_VOLUME_RATIO_CEIL = 1.5         # 1.5x avg scores 1

# Foreign flow settings
RULE_FOREIGN_FLOOR = -1_000_000_000  # -1B IDR scores 0
RULE_FOREIGN_CEIL = 10_000_000_000   # +10B IDR scores 1

# Slope score mapping
RULE_SLOPE_SCORE_FLOOR = -0.001
RULE_SLOPE_SCORE_CEIL = 0.005

# Score weights (sum = 100)
RULE_SCORE_WEIGHT_MOMENTUM = 32      # Recent price action
RULE_SCORE_WEIGHT_SLOPE = 23         # MA50 trend direction
RULE_SCORE_WEIGHT_DIST50 = 18        # Position above MA50
RULE_SCORE_WEIGHT_VOLUME = 17        # Volume confirmation
RULE_SCORE_WEIGHT_FOREIGN = 10       # Foreign flow signal
RULE_SCORE_DIST50_CAP = 0.10         # Dist50 scoring cap
```

---

## Data Collection

### Daily Data Sources

| Source | Command | Purpose |
|--------|---------|---------|
| Daily OHLCV | `collect-data` | Price, volume, value |
| Foreign Flow | `collect-foreign` | Net foreign buy/sell |
| Movers | `collect-movers` | Top value/volume/frequency lists |
| Broker Summary | `collect-broker` | Broker activity (optional) |
| Insider Trades | `collect-insider` | Insider activity (optional) |

### Foreign Flow Collection

Foreign flow data is collected from the movers API endpoints:
- `net_foreign_buy` - Top 50 stocks with highest foreign buying
- `net_foreign_sell` - Top 50 stocks with highest foreign selling

```bash
# Collect today's foreign flow
python main.py collect-foreign

# Backfill historical (slow, ~0.5s per stock)
python main.py collect-foreign --backfill --start 2025-12-01 --end 2026-01-23
```

---

## Code Implementation

### File Structure

```
models/
└── rule_based.py          # Main strategy implementation
    ├── RuleBasedPredictor  # Main class
    ├── _precompute_indicators()
    ├── _signal_for_index()
    ├── _score_candidate()
    ├── predict()
    └── backtest_last_days()
```

### Key Methods

#### 1. `_precompute_indicators(df: DataFrame) -> DataFrame`

Pre-computes all technical indicators for efficiency:

```python
def _precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma50"] = df["close"].rolling(window=RULE_MA_SLOW).mean()
    df["slope50"] = (df["ma50"] - df["ma50"].shift(RULE_SLOPE_LOOKBACK)) / df["ma50"].shift(RULE_SLOPE_LOOKBACK)
    df["dist50"] = (df["close"] - df["ma50"]) / df["ma50"]
    df["momentum"] = df["close"].pct_change(periods=RULE_MOMENTUM_PERIOD)
    df["volume_ma"] = df["volume"].rolling(window=RULE_VOLUME_MA_PERIOD).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    return df
```

#### 2. `_signal_for_index(df: DataFrame, idx: int) -> Optional[dict]`

Evaluates a single stock on a specific date:

```python
def _signal_for_index(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
    # Read pre-computed indicators
    dist50 = df["dist50"].iloc[idx]
    momentum = df["momentum"].iloc[idx]
    volume_ratio = df["volume_ratio"].iloc[idx]
    foreign_net = df["foreign_net"].iloc[idx] if "foreign_net" in df.columns else None

    # Hard filters
    if dist50 < RULE_DIST50_MIN:      # Must be above MA50
        return None
    if dist50 > RULE_DIST50_MAX:      # Not too extended
        return None
    if slope50_t < RULE_SLOPE_FLAT_MIN:  # MA50 not falling
        return None

    # Calculate score (includes foreign flow)
    score = self._score_candidate(momentum, slope50_t, dist50, volume_ratio, foreign_net)

    return {"score": score, "momentum": momentum, ...}
```

#### 3. `_score_candidate(...) -> float`

Calculates the composite score with all 5 components:

```python
def _score_candidate(self, momentum, slope50, dist50, volume_ratio, foreign_net=None) -> float:
    momentum_score = self._clamp(...)
    slope_score = self._clamp(...)
    dist50_score = self._clamp(...)
    volume_score = self._clamp(...)

    # Foreign flow score (weight 0 if no data)
    if foreign_net is not None and not pd.isna(foreign_net):
        foreign_score = self._clamp(
            (foreign_net - RULE_FOREIGN_FLOOR) / (RULE_FOREIGN_CEIL - RULE_FOREIGN_FLOOR)
        )
    else:
        foreign_score = 0.0

    foreign_weight = RULE_SCORE_WEIGHT_FOREIGN if foreign_net is not None and not pd.isna(foreign_net) else 0
    raw_score = (
        momentum_score * RULE_SCORE_WEIGHT_MOMENTUM
        + slope_score * RULE_SCORE_WEIGHT_SLOPE
        + dist50_score * RULE_SCORE_WEIGHT_DIST50
        + volume_score * RULE_SCORE_WEIGHT_VOLUME
        + foreign_score * foreign_weight
    )
    total_weight = (
        RULE_SCORE_WEIGHT_MOMENTUM
        + RULE_SCORE_WEIGHT_SLOPE
        + RULE_SCORE_WEIGHT_DIST50
        + RULE_SCORE_WEIGHT_VOLUME
        + foreign_weight
    )
    return raw_score / total_weight
```

---

## Usage

### Daily Prediction

```bash
# Generate today's top 5 picks
python main.py predict --top 5

# Generate picks with component scores
python main.py predict-scores --top 5

# With Telegram notification
python main.py predict --top 5 --telegram
```

### Backtesting

```bash
# Last 30 trading days
python main.py backtest-rules --days 30 --top 5

# Last 60 trading days
python main.py backtest-rules --days 60 --top 5
```

### Full Daily Workflow

```bash
# Collect data + foreign flow + predict + notify
python main.py daily
```

### Data Coverage Verification

```bash
# Verify data coverage for a date range
python main.py verify-range --start 2025-01-20 --end 2026-01-23
```

---

## Performance

Based on backtest (30 trading days, Dec 2025 - Jan 2026):

| Configuration | Precision@5 |
|--------------|-------------|
| Baseline (no filters) | 21.5% |
| + Movers filter | 25.83% |
| + Foreign flow (10%) | **27.50%** |

| Metric | Value |
|--------|-------|
| **Current Precision@5** | **27.50%** |
| Baseline (random) | 10.2% |
| Edge vs Baseline | **+17.3%** |

The strategy is approximately **2.7x better** than random selection at identifying stocks that gain ≥5% the next day.

---

## Limitations & Notes

1. **Foreign Flow Coverage**: Only ~1% of records have foreign flow data. Stocks without data get zero foreign weight.

2. **No Stop-Loss**: Strategy only handles entry ranking, not exit/risk management.

3. **No Position Sizing**: All picks are equal-weighted in backtest.

4. **T+2 Settlement**: IDX has T+2 settlement; not enforced in current backtest.

5. **Holiday Calendar**: Only weekends are filtered; IDX holidays (Eid, Nyepi, etc.) not yet fully implemented.

---

## Future Improvements

- [x] ~~Add market regime filter (movers-based)~~ ✅ Implemented
- [x] ~~Add foreign flow signal~~ ✅ Implemented
- [ ] Add IDX holiday calendar
- [ ] Implement T+2 settlement cooldown in backtest
- [ ] Add sector diversification filter
- [ ] Add stop-loss tracking with intraday data
- [ ] Test alternative momentum periods (3-day, 10-day)
