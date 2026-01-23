Quants-API
==========

Indonesian Stock Market rule-based daily ranking system using MA50 + Momentum + Foreign Flow.

Overview
--------
- **Strategy**: MA50 trend filter + momentum + foreign flow scoring
- **Data sources**: Datasaham API (daily OHLCV, foreign flow, movers)
- **Storage**: SQLite (`database/quants.db`)
- **Output**: Top 5 ranked stock picks after market close
- **Performance**: 27.50% precision (2.7x better than random)

Quick Start
-----------
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize database:
   ```bash
   python main.py init
   ```

3. Collect stock universe:
   ```bash
   python main.py collect-stocks
   ```

4. Load historical data:
   ```bash
   python main.py load-historical --days 365
   ```

5. Backfill movers data:
   ```bash
   python main.py collect-movers --backfill --start 2025-01-20 --end 2026-01-23 --top 50
   ```

6. Generate ranked picks:
   ```bash
   python main.py predict --top 5
   ```

Daily Workflow
--------------
Run after market close (4:45 PM WIB):
```bash
python daily_run.py
```

This will:
1. Collect daily OHLCV data
2. Collect foreign flow data
3. Update prediction actuals
4. Generate next-day ranked picks
5. Send Telegram summary (if configured)

Strategy Overview
-----------------
The strategy uses 5 scoring components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Momentum | 32% | 5-day price return |
| Slope | 23% | MA50 trend direction |
| Dist50 | 18% | Position above MA50 |
| Volume | 17% | Volume vs 20-day avg |
| Foreign | 10% | Net foreign flow |

**Filters:**
- Movers filter: Only trade stocks in top value/volume/frequency lists
- Above MA50: Price must be above 50-day MA
- Not overextended: Max 15% above MA50
- Slope not falling: MA50 must not be declining sharply

See `docs/STRATEGY.md` for full documentation.

CLI Commands
------------
```bash
# Data collection
python main.py collect-stocks          # Collect stock universe
python main.py collect-data            # Collect daily OHLCV
python main.py collect-foreign         # Collect foreign flow (today)
python main.py collect-foreign --backfill  # Backfill foreign flow
python main.py collect-movers          # Collect movers lists
python main.py load-historical --days 365  # Load historical data

# Predictions
python main.py predict --top 5         # Generate top 5 picks
python main.py predict --telegram      # With Telegram notification

# Backtesting
python main.py backtest-rules --days 30    # 30-day backtest
python main.py backtest-rules --days 60    # 60-day backtest

# Utilities
python main.py verify                  # Verify data and system
python main.py telegram-test           # Test Telegram notification
python main.py daily                   # Full daily workflow
```

Configuration
-------------
Edit `config.py`:
- `TOP_PICKS_COUNT`: Number of stocks to pick (default: 5)
- `TOP_GAINER_THRESHOLD`: Target return threshold (default: 5%)
- `MOVERS_FILTER_ENABLED`: Enable movers filter (default: True)
- `RULE_SCORE_WEIGHT_*`: Scoring component weights
- See `docs/STRATEGY.md` for all parameters

Performance
-----------
Based on 30-day backtest (Dec 2025 - Jan 2026):

| Metric | Value |
|--------|-------|
| Precision@5 | 27.50% |
| Random baseline | 10.2% |
| Edge | +17.3% (2.7x) |

Repository Notes
----------------
- `database/quants.db` is tracked with Git LFS. After cloning, run `git lfs pull`.
- Foreign flow data is collected daily and accumulates over time.
- The movers filter significantly improves precision (+4%).
