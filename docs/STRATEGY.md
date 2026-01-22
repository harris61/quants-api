# Trading Strategy Documentation

## Overview

**Strategy Name:** MA50 + Momentum Daily Ranking
**Model Version:** `rule_ma50_v2`
**Target:** Indonesian Stock Exchange (IDX) equities
**Objective:** Identify stocks likely to gain ≥5% in the next trading day

---

## Strategy Logic

### Core Concept

The strategy combines **trend-following** (MA50) with **momentum** to rank stocks daily. It filters for stocks in uptrends and scores them by recent price momentum, trend strength, and volume confirmation.

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

Stocks passing the filters are scored (0-1 scale) using four components:

| Component | Weight | Range | Config Parameters |
|-----------|--------|-------|-------------------|
| **Momentum** | 35% | 5-day return: -5% to +10% | `RULE_MOMENTUM_FLOOR`, `RULE_MOMENTUM_CEIL` |
| **Slope** | 25% | MA50 slope: -0.1% to +0.5% | `RULE_SLOPE_SCORE_FLOOR`, `RULE_SLOPE_SCORE_CEIL` |
| **Dist50** | 20% | Distance above MA50: 0% to 10% | `RULE_SCORE_DIST50_CAP` |
| **Volume** | 20% | Volume ratio: 0.5x to 1.5x avg | `RULE_VOLUME_RATIO_FLOOR`, `RULE_VOLUME_RATIO_CEIL` |

**Score Formula:**
```
score = (momentum_score × 35 + slope_score × 25 + dist50_score × 20 + volume_score × 20) / 100
```

Each component is clamped to [0, 1]:
```python
momentum_score = clamp((momentum - (-0.05)) / (0.10 - (-0.05)))  # -5%→0, +10%→1
slope_score    = clamp((slope50 - (-0.001)) / (0.005 - (-0.001)))
dist50_score   = clamp(dist50 / 0.10)                            # 0%→0, 10%→1
volume_score   = clamp((volume_ratio - 0.5) / (1.5 - 0.5))       # 0.5x→0, 1.5x→1
```

### Daily Output

The strategy outputs the **top 5 stocks** ranked by score, with:
- Symbol
- Score (0-1)
- 5-day momentum (%)
- Volume ratio (x avg)

---

## Configuration Reference

All parameters are in `config.py`:

```python
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

# Slope score mapping
RULE_SLOPE_SCORE_FLOOR = -0.001
RULE_SLOPE_SCORE_CEIL = 0.005

# Score weights (sum = 100)
RULE_SCORE_WEIGHT_MOMENTUM = 35
RULE_SCORE_WEIGHT_SLOPE = 25
RULE_SCORE_WEIGHT_DIST50 = 20
RULE_SCORE_WEIGHT_VOLUME = 20
RULE_SCORE_DIST50_CAP = 0.10

# Output settings
TOP_PICKS_COUNT = 5                  # Top N stocks per day
TOP_GAINER_THRESHOLD = 0.05          # 5% target for "correct" prediction
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
    close = df["close"].iloc[idx]
    ma50_t = df["ma50"].iloc[idx]
    slope50_t = df["slope50"].iloc[idx]
    dist50 = df["dist50"].iloc[idx]
    momentum = df["momentum"].iloc[idx]
    volume_ratio = df["volume_ratio"].iloc[idx]

    # Hard filters
    if dist50 < RULE_DIST50_MIN:      # Must be above MA50
        return None
    if dist50 > RULE_DIST50_MAX:      # Not too extended
        return None
    if slope50_t < RULE_SLOPE_FLAT_MIN:  # MA50 not falling
        return None

    # Calculate score
    score = self._score_candidate(momentum, slope50_t, dist50, volume_ratio)

    return {
        "score": score,
        "momentum": momentum,
        "dist50": dist50,
        "slope50": slope50_t,
        "volume_ratio": volume_ratio,
    }
```

#### 3. `_score_candidate(...) -> float`

Calculates the composite score:

```python
def _score_candidate(self, momentum, slope50, dist50, volume_ratio) -> float:
    momentum_score = self._clamp(
        (momentum - RULE_MOMENTUM_FLOOR) / (RULE_MOMENTUM_CEIL - RULE_MOMENTUM_FLOOR)
    )
    slope_score = self._clamp(
        (slope50 - RULE_SLOPE_SCORE_FLOOR) / (RULE_SLOPE_SCORE_CEIL - RULE_SLOPE_SCORE_FLOOR)
    )
    dist50_score = self._clamp(dist50 / RULE_SCORE_DIST50_CAP)
    volume_score = self._clamp(
        (volume_ratio - RULE_VOLUME_RATIO_FLOOR) / (RULE_VOLUME_RATIO_CEIL - RULE_VOLUME_RATIO_FLOOR)
    )

    total_weight = (RULE_SCORE_WEIGHT_MOMENTUM + RULE_SCORE_WEIGHT_SLOPE
                    + RULE_SCORE_WEIGHT_DIST50 + RULE_SCORE_WEIGHT_VOLUME)

    raw_score = (
        momentum_score * RULE_SCORE_WEIGHT_MOMENTUM
        + slope_score * RULE_SCORE_WEIGHT_SLOPE
        + dist50_score * RULE_SCORE_WEIGHT_DIST50
        + volume_score * RULE_SCORE_WEIGHT_VOLUME
    )
    return raw_score / total_weight
```

#### 4. `predict(symbols, top_k, save_to_db) -> DataFrame`

Main prediction method:

```python
def predict(self, symbols=None, top_k=None, save_to_db=True) -> pd.DataFrame:
    top_k = top_k or TOP_PICKS_COUNT
    symbols = symbols or self._get_active_symbols()

    # Load and pre-compute indicators for all stocks
    stock_data = self._load_all_stock_data_batch(symbols)

    results = []
    for symbol in symbols:
        df = stock_data.get(symbol)
        signal = self._latest_signal(df)
        if signal:
            results.append({
                "symbol": symbol,
                "probability": signal["score"],
                "momentum": signal["momentum"],
                "dist50": signal["dist50"],
                "slope50": signal["slope50"],
                "volume_ratio": signal["volume_ratio"],
            })

    # Sort by score and return top K
    results_df = pd.DataFrame(results).sort_values("probability", ascending=False)
    return results_df.head(top_k)
```

#### 5. `backtest_last_days(days, top_k) -> DataFrame`

Walk-forward backtest:

```python
def backtest_last_days(self, days=30, top_k=None) -> pd.DataFrame:
    # Load data with sufficient buffer for MA50 warmup
    buffer_days = int((RULE_MA_SLOW + RULE_SLOPE_LOOKBACK + days) * 1.5) + 20

    # Get trading days only (exclude weekends)
    all_dates = [d for d in db_dates if is_trading_day(d)]
    eval_dates = all_dates[-days:]

    for eval_date in eval_dates:
        # Get signals for all stocks on this date
        # Take top K by score
        # Check if next-day return >= threshold
        # Record precision

    return results_df
```

---

## Usage

### Daily Prediction

```bash
# Generate today's top 5 picks
python main.py predict --top 5

# With Telegram notification
python main.py predict --top 5 --telegram
```

### Backtesting

```bash
# Last 30 trading days
python main.py backtest-rules --days 30 --top 5

# Last 60 trading days with top 10
python main.py backtest-rules --days 60 --top 10
```

### Full Daily Workflow

```bash
# Collect data + predict + notify
python main.py daily
```

---

## Performance

Based on backtest (30 trading days, Dec 2025 - Jan 2026):

| Metric | Value |
|--------|-------|
| Precision@5 | **21.5%** |
| Precision@10 | 18.0% |
| Baseline (random) | 10.2% |
| Edge vs Baseline | **+8-11%** |

The strategy is approximately **2x better** than random selection at identifying stocks that gain ≥5% the next day.

---

## Limitations & Notes

1. **Data Quality**: Backtest assumes clean data. Weekends/holidays in database are filtered but may still affect results.

2. **No Stop-Loss**: Strategy only handles entry ranking, not exit/risk management.

3. **No Position Sizing**: All picks are equal-weighted in backtest.

4. **T+2 Settlement**: IDX has T+2 settlement; not enforced in current backtest.

5. **Holiday Calendar**: Only weekends are filtered; IDX holidays (Eid, Nyepi, etc.) not yet implemented.

---

## Future Improvements

- [ ] Add IDX holiday calendar
- [ ] Implement T+2 settlement cooldown in backtest
- [ ] Add sector diversification filter
- [ ] Test alternative momentum periods (3-day, 10-day)
- [ ] Add market regime filter (IHSG trend)
