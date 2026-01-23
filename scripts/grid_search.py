"""
Parameter Grid Search for Rule-Based Strategy
Run: python scripts/grid_search.py

Testing: Additional Filters (Min Value, Gap Filter)
"""

import sys
import os
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

import config
from database import init_db, session_scope, Stock, DailyPrice
from models.rule_based import RuleBasedPredictor


# ==================== TEST CONFIGURATIONS ====================

# Min daily value (IDR) - filter illiquid stocks
MIN_VALUE_OPTIONS = [0, 1_000_000_000, 5_000_000_000, 10_000_000_000]  # 0, 1B, 5B, 10B

# Max gap from previous day - filter stocks that gapped too much
MAX_GAP_OPTIONS = [1.0, 0.10, 0.07, 0.05]  # 1.0 = no filter, 0.10 = max 10% gap

BACKTEST_DAYS = 30
TOP_PICKS = 3


def backtest_with_filters(min_value: float, max_gap: float) -> tuple:
    """Run backtest with additional filters applied."""

    with session_scope() as session:
        # Get date range
        max_date = session.query(DailyPrice.date).order_by(DailyPrice.date.desc()).first()
        if not max_date:
            return 0.0, 0, 0

        max_date = max_date[0]

        # Get all trading days
        dates = session.query(DailyPrice.date).distinct().order_by(DailyPrice.date.desc()).limit(BACKTEST_DAYS + 5).all()
        dates = sorted([d[0] for d in dates])

        if len(dates) < BACKTEST_DAYS + 1:
            return 0.0, 0, 0

        eval_dates = dates[-(BACKTEST_DAYS + 1):-1]  # Exclude last day (no next day return)

        total_correct = 0
        total_picks = 0

        predictor = RuleBasedPredictor()

        for eval_date in eval_dates:
            # Get next trading day
            next_dates = [d for d in dates if d > eval_date]
            if not next_dates:
                continue
            next_date = next_dates[0]

            # Get predictions for eval_date
            with redirect_stdout(io.StringIO()):
                df = predictor.predict(top_k=20, save_to_db=False)  # Get more candidates

            if df is None or df.empty:
                continue

            # Apply additional filters
            filtered_symbols = []

            for _, row in df.iterrows():
                symbol = row['symbol']

                # Get price data for eval_date
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    continue

                price_today = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == stock.id,
                    DailyPrice.date == eval_date
                ).first()

                price_prev = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == stock.id,
                    DailyPrice.date < eval_date
                ).order_by(DailyPrice.date.desc()).first()

                if not price_today:
                    continue

                # Filter 1: Min value
                if min_value > 0 and (price_today.value or 0) < min_value:
                    continue

                # Filter 2: Max gap
                if max_gap < 1.0 and price_prev and price_prev.close > 0:
                    gap = abs(price_today.open - price_prev.close) / price_prev.close
                    if gap > max_gap:
                        continue

                filtered_symbols.append(symbol)

                if len(filtered_symbols) >= TOP_PICKS:
                    break

            # Check results for filtered picks
            for symbol in filtered_symbols[:TOP_PICKS]:
                stock = session.query(Stock).filter(Stock.symbol == symbol).first()
                if not stock:
                    continue

                price_today = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == stock.id,
                    DailyPrice.date == eval_date
                ).first()

                price_next = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == stock.id,
                    DailyPrice.date == next_date
                ).first()

                if price_today and price_next and price_today.close > 0:
                    ret = (price_next.close - price_today.close) / price_today.close
                    if ret >= config.TOP_GAINER_THRESHOLD:
                        total_correct += 1
                    total_picks += 1

        precision = total_correct / total_picks if total_picks > 0 else 0.0
        return precision, total_correct, total_picks


def run_grid_search():
    init_db()

    print("Grid Search: Testing Additional Filters")
    print(f"Min Value Options: {[f'{v/1e9:.0f}B' if v > 0 else 'None' for v in MIN_VALUE_OPTIONS]}")
    print(f"Max Gap Options: {[f'{g*100:.0f}%' if g < 1.0 else 'None' for g in MAX_GAP_OPTIONS]}")
    print(f"Backtest: {BACKTEST_DAYS} days, top {TOP_PICKS} picks")
    print("-" * 80)

    results = []

    # Test combinations
    combinations = [(v, g) for v in MIN_VALUE_OPTIONS for g in MAX_GAP_OPTIONS]

    for min_value, max_gap in tqdm(combinations, desc="Testing"):
        try:
            precision, correct, total = backtest_with_filters(min_value, max_gap)

            results.append({
                "min_value": f"{min_value/1e9:.0f}B" if min_value > 0 else "None",
                "max_gap": f"{max_gap*100:.0f}%" if max_gap < 1.0 else "None",
                "precision": precision,
                "precision_pct": f"{precision * 100:.2f}%",
                "correct": correct,
                "total": total,
            })

        except Exception as e:
            tqdm.write(f"Error: {e}")
            results.append({
                "min_value": f"{min_value/1e9:.0f}B" if min_value > 0 else "None",
                "max_gap": f"{max_gap*100:.0f}%" if max_gap < 1.0 else "None",
                "precision": 0.0,
                "precision_pct": "ERROR",
                "correct": 0,
                "total": 0,
            })

    df = pd.DataFrame(results)
    df = df.sort_values("precision", ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(__file__), f"grid_search_filters_{timestamp}.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("ALL CONFIGURATIONS (sorted by precision):")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    # Best config
    best = df.iloc[0]
    print("\n" + "-" * 80)
    print("BEST CONFIGURATION:")
    print(f"  Min Value: {best['min_value']}")
    print(f"  Max Gap: {best['max_gap']}")
    print(f"  Precision: {best['precision_pct']} ({best['correct']}/{best['total']})")


if __name__ == "__main__":
    run_grid_search()
