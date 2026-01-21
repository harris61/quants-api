Quants-API
==========

Indonesian Stock Market ML pipeline for daily top-gainer prediction (intraday open -> close).

Overview
--------
- Data sources: Datasaham API (daily OHLCV, broker summary, insider, intraday, movers)
- Storage: SQLite (`database/quants.db`)
- Model: LightGBM binary classifier
- Target: next trading day intraday return >= `TOP_GAINER_THRESHOLD`

Quick Start
-----------
1) Install dependencies:
   - `python -m pip install -r requirements.txt`
2) Initialize database:
   - `python main.py init`
3) Collect stock universe (equity tickers only):
   - `python main.py collect-stocks`
4) Load historical data:
   - `python main.py load-historical --days 365`
5) (Optional) Backfill movers from existing daily data:
   - `python main.py collect-movers --backfill --start 2025-01-20 --end 2026-01-20 --top 50`
6) Train model:
   - `python main.py train --name my_model`
7) Predict top picks:
   - `python main.py predict --top 10`

Daily Workflow
--------------
Run after market close:
- `python daily_run.py`

This will:
1) Collect daily data, broker summaries, intraday, and movers
2) Update prediction actuals
3) Generate next-day predictions
4) Send Telegram summary (if configured)

Backtesting
-----------
Walk-forward backtest with trade simulation:
- `python main.py backtest --start 2025-10-22 --end 2026-01-20 --save`

Notes:
- Prediction target is next-day intraday return (open -> close).
- Backtest enforces T+3 trading-day cooldown between trades.
- Results are saved to `models/saved/*.csv`.

Movers Data
-----------
Movers are stored in `daily_movers` and used as features.
To backfill from DB history:
- `python main.py collect-movers --backfill --start YYYY-MM-DD --end YYYY-MM-DD --top 50`

Configuration
-------------
Edit `config.py`:
- `TOP_GAINER_THRESHOLD`: intraday return threshold
- `EQUITY_SYMBOL_REGEX`: filter for equity universe
- Feature toggles: `INCLUDE_BROKER_FEATURES`, `INCLUDE_INSIDER_FEATURES`, `INCLUDE_INTRADAY_FEATURES`, `INCLUDE_MOVER_FEATURES`
- Movers collection: `MOVERS_COLLECTION_ENABLED`

Repository Notes
----------------
- `database/quants.db` is tracked with Git LFS. After cloning, run `git lfs pull`.
- The analysis notebook is excluded from version control.
