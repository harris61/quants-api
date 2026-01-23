"""
Parameter Grid Search for Rule-Based Strategy
Run: python scripts/grid_search.py

Testing: Additional Filters (Min Value, Gap Filter)
Uses existing backtest, then applies filters to results
"""

import sys
import os
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from tqdm import tqdm

import config
from database import init_db, session_scope, Stock, DailyPrice
from models.rule_based import RuleBasedPredictor


# ==================== TEST CONFIGURATIONS ====================

MIN_VALUE_OPTIONS = [0, 1_000_000_000, 5_000_000_000, 10_000_000_000]
MAX_GAP_OPTIONS = [1.0, 0.10, 0.07, 0.05]

BACKTEST_DAYS = 30
TOP_PICKS = 3


def run_grid_search():
    init_db()

    print("Grid Search: Testing Additional Filters")
    print(f"Min Value: {[f'{v/1e9:.0f}B' if v > 0 else 'None' for v in MIN_VALUE_OPTIONS]}")
    print(f"Max Gap: {[f'{g*100:.0f}%' if g < 1.0 else 'None' for g in MAX_GAP_OPTIONS]}")
    print("-" * 80)

    # Run backtest once with expanded candidates
    print("\nRunning base backtest (top 20 candidates)...")
    predictor = RuleBasedPredictor()

    with redirect_stdout(io.StringIO()):
        base_df = predictor.backtest_last_days(days=BACKTEST_DAYS, top_k=20, save_csv=False)

    if base_df is None or base_df.empty:
        print("No backtest results.")
        return

    # Parse picks from backtest results
    all_picks = []
    for _, row in base_df.iterrows():
        date = row["date"]
        for i in range(1, 21):
            col = f"stockpick_{i}"
            if col not in row or not row[col] or "/" not in str(row[col]):
                continue
            val = str(row[col])
            symbol = val.split("/")[0]
            ret_str = val.split("/")[1].replace("%", "").replace("+", "")
            try:
                ret = float(ret_str) / 100
            except:
                continue
            all_picks.append({"date": date, "symbol": symbol, "return": ret, "rank": i})

    picks_df = pd.DataFrame(all_picks)
    print(f"Loaded {len(picks_df)} picks from {len(base_df)} days")

    # Load price data for filtering
    print("Loading price data...")
    with session_scope() as session:
        price_data = {}
        for symbol in picks_df["symbol"].unique():
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                continue
            prices = session.query(DailyPrice).filter(DailyPrice.stock_id == stock.id).all()
            price_data[symbol] = {p.date: {"value": p.value or 0, "open": p.open or 0, "close": p.close or 0} for p in prices}

    # Test filter combinations
    print("Testing combinations...")
    results = []

    for min_value, max_gap in tqdm([(v, g) for v in MIN_VALUE_OPTIONS for g in MAX_GAP_OPTIONS], desc="Testing"):
        correct = 0
        total = 0

        for date in picks_df["date"].unique():
            day_picks = picks_df[picks_df["date"] == date].sort_values("rank")
            count = 0

            for _, pick in day_picks.iterrows():
                if count >= TOP_PICKS:
                    break

                symbol = pick["symbol"]
                if symbol not in price_data or date not in price_data[symbol]:
                    continue

                p = price_data[symbol][date]

                # Min value filter
                if min_value > 0 and p["value"] < min_value:
                    continue

                # Gap filter
                if max_gap < 1.0:
                    dates_list = sorted(price_data[symbol].keys())
                    idx = dates_list.index(date) if date in dates_list else -1
                    if idx > 0:
                        prev = price_data[symbol][dates_list[idx - 1]]
                        if prev["close"] > 0:
                            gap = abs(p["open"] - prev["close"]) / prev["close"]
                            if gap > max_gap:
                                continue

                count += 1
                total += 1
                if pick["return"] >= config.TOP_GAINER_THRESHOLD:
                    correct += 1

        precision = correct / total if total > 0 else 0
        results.append({
            "min_value": f"{min_value/1e9:.0f}B" if min_value > 0 else "None",
            "max_gap": f"{max_gap*100:.0f}%" if max_gap < 1.0 else "None",
            "precision": precision,
            "precision_pct": f"{precision*100:.2f}%",
            "correct": correct,
            "total": total,
        })

    df = pd.DataFrame(results).sort_values("precision", ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(df.to_string(index=False))

    baseline = df[(df["min_value"] == "None") & (df["max_gap"] == "None")].iloc[0]
    best = df.iloc[0]

    print(f"\nBaseline: {baseline['precision_pct']} ({baseline['correct']}/{baseline['total']})")
    print(f"Best: {best['min_value']} / {best['max_gap']} = {best['precision_pct']} ({best['correct']}/{best['total']})")


if __name__ == "__main__":
    run_grid_search()
