"""
Inspect Actual Stock Picks
Run: python scripts/inspect_picks.py

Shows detailed breakdown of which stocks are being selected and why.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

from database import init_db, session_scope, Stock, DailyPrice, Prediction
from models.rule_based import RuleBasedPredictor
import config


def inspect_picks():
    init_db()

    print("=" * 80)
    print("INSPECTING STOCK PICKS")
    print("=" * 80)

    # Get recent predictions
    with session_scope() as session:
        recent_preds = session.query(
            Prediction, Stock
        ).join(Stock).filter(
            Prediction.rank <= config.TOP_PICKS_COUNT
        ).order_by(
            Prediction.prediction_date.desc(),
            Prediction.rank
        ).limit(90).all()  # Last ~30 days * 3 picks

        if not recent_preds:
            print("No predictions found. Run 'python main.py predict' first.")
            return

        # Analyze picks
        print(f"\nLast {len(recent_preds)} predictions analyzed:\n")

        # Count stock frequency
        stock_counts = Counter()
        stock_correct = Counter()
        stock_total = Counter()
        stock_sectors = {}

        for pred, stock in recent_preds:
            stock_counts[stock.symbol] += 1
            stock_sectors[stock.symbol] = stock.sector_name or "Unknown"
            if pred.actual_return is not None:
                stock_total[stock.symbol] += 1
                if pred.is_top_gainer:
                    stock_correct[stock.symbol] += 1

        # Most frequent picks
        print("-" * 80)
        print("MOST FREQUENTLY PICKED STOCKS:")
        print("-" * 80)
        print(f"{'Symbol':<8} {'Count':<8} {'Sector':<25} {'Hit Rate':<12}")
        print("-" * 80)

        for symbol, count in stock_counts.most_common(15):
            sector = stock_sectors.get(symbol, "Unknown")[:24]
            total = stock_total.get(symbol, 0)
            correct = stock_correct.get(symbol, 0)
            hit_rate = f"{correct}/{total} ({correct/total*100:.0f}%)" if total > 0 else "N/A"
            print(f"{symbol:<8} {count:<8} {sector:<25} {hit_rate:<12}")

        # Sector distribution
        print("\n" + "-" * 80)
        print("SECTOR DISTRIBUTION:")
        print("-" * 80)

        sector_counts = Counter()
        for symbol, count in stock_counts.items():
            sector = stock_sectors.get(symbol, "Unknown")
            sector_counts[sector] += count

        for sector, count in sector_counts.most_common(10):
            pct = count / len(recent_preds) * 100
            print(f"  {sector[:30]:<32} {count:>4} picks ({pct:.1f}%)")

        # Get latest day's picks with details
        print("\n" + "-" * 80)
        print("LATEST PICKS WITH DETAILS:")
        print("-" * 80)

        latest_date = recent_preds[0][0].prediction_date
        latest_picks = [(p, s) for p, s in recent_preds if p.prediction_date == latest_date]

        print(f"\nDate: {latest_date}\n")

        for pred, stock in latest_picks:
            print(f"Rank {pred.rank}: {stock.symbol}")
            print(f"  Sector: {stock.sector_name or 'Unknown'}")
            print(f"  Score: {pred.probability:.4f}")

            # Get price data for context
            price = session.query(DailyPrice).filter(
                DailyPrice.stock_id == stock.id,
                DailyPrice.date <= latest_date
            ).order_by(DailyPrice.date.desc()).first()

            if price:
                print(f"  Close: {price.close:,.0f}")
                print(f"  Volume: {price.volume:,.0f}")
                print(f"  Value: {price.value:,.0f}")
                if price.foreign_net:
                    print(f"  Foreign Net: {price.foreign_net:,.0f}")

            if pred.actual_return is not None:
                result = "✓ HIT" if pred.is_top_gainer else "✗ MISS"
                print(f"  Result: {pred.actual_return*100:+.2f}% {result}")
            print()

    # Run fresh prediction to show component scores
    print("-" * 80)
    print("CURRENT PREDICTION WITH COMPONENT SCORES:")
    print("-" * 80)

    predictor = RuleBasedPredictor()

    # Get predictions with scores
    try:
        df = predictor.predict(top_k=5, save_to_db=False, include_components=True)

        if df is not None and not df.empty:
            print(f"\nTop {len(df)} candidates:\n")

            # Display columns
            cols = ['symbol', 'probability', 'momentum', 'dist50', 'volume_ratio']
            if 'slope50' in df.columns:
                cols.insert(3, 'slope50')

            display_df = df[cols].copy()
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.4f}")
            display_df['momentum'] = display_df['momentum'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['dist50'] = display_df['dist50'].apply(lambda x: f"{x*100:.2f}%")
            if 'slope50' in display_df.columns:
                display_df['slope50'] = display_df['slope50'].apply(lambda x: f"{x*100:.3f}%")
            display_df['volume_ratio'] = display_df['volume_ratio'].apply(lambda x: f"{x:.2f}x")

            print(display_df.to_string(index=False))
    except Exception as e:
        print(f"Could not generate fresh prediction: {e}")

    print("\n" + "=" * 80)
    print("INSIGHTS:")
    print("=" * 80)
    print("- Check if same stocks keep appearing (concentration risk)")
    print("- Check sector distribution (diversification)")
    print("- Compare hit rates per stock (consistent performers)")
    print()


if __name__ == "__main__":
    inspect_picks()
