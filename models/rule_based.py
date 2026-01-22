"""
Rule-Based Predictor - MA20/MA50 daily ranking (long-only)
"""

from datetime import datetime, timedelta
from typing import List, Optional
import re

import pandas as pd

from config import (
    RULE_BASED_MODEL_NAME,
    RULE_MA_FAST,
    RULE_MA_SLOW,
    RULE_SLOPE_LOOKBACK,
    RULE_DIST20_ENTRY,
    RULE_DIST20_OVEREXTENDED,
    RULE_DIST50_FALLING_KNIFE,
    RULE_SLOPE_FLAT_MIN,
    RULE_SLOPE_SCORE_FLOOR,
    RULE_SLOPE_SCORE_CEIL,
    RULE_SCORE_WEIGHT_PROX,
    RULE_SCORE_WEIGHT_SLOPE,
    RULE_SCORE_WEIGHT_DIST50,
    RULE_SCORE_WEIGHT_RECLAIM,
    RULE_SCORE_DIST50_CAP,
    TOP_PICKS_COUNT,
    TOP_GAINER_THRESHOLD,
    MOVERS_FILTER_ENABLED,
    MOVERS_FILTER_TYPES,
    EQUITY_SYMBOL_REGEX,
)
from database import session_scope, Stock, DailyPrice, Prediction


class RuleBasedPredictor:
    """Generate daily ranked picks using MA20/MA50 rules."""

    def __init__(self) -> None:
        self.model_name = RULE_BASED_MODEL_NAME
        self.model_type = "ranking"
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def _clamp(self, value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(value, max_value))

    def _is_equity_symbol(self, symbol: str) -> bool:
        if not symbol:
            return False
        return bool(self._equity_pattern.match(symbol.upper()))

    def _get_active_symbols(self) -> List[str]:
        with session_scope() as session:
            rows = session.query(Stock.symbol).filter(Stock.is_active == True).all()
        symbols = [r[0] for r in rows]
        return [s for s in symbols if self._is_equity_symbol(s)]

    def _load_all_stock_data_batch(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> dict:
        if not symbols:
            return {}

        with session_scope() as session:
            query = session.query(
                Stock.symbol,
                DailyPrice.date,
                DailyPrice.open,
                DailyPrice.high,
                DailyPrice.low,
                DailyPrice.close,
                DailyPrice.volume,
                DailyPrice.value,
                DailyPrice.frequency,
                DailyPrice.foreign_buy,
                DailyPrice.foreign_sell,
                DailyPrice.foreign_net,
            ).join(DailyPrice, DailyPrice.stock_id == Stock.id).filter(
                Stock.symbol.in_([s.upper() for s in symbols])
            )

            if start_date:
                query = query.filter(DailyPrice.date >= start_date)
            if end_date:
                query = query.filter(DailyPrice.date <= end_date)

            query = query.order_by(Stock.symbol, DailyPrice.date)
            rows = query.all()

        if not rows:
            return {}

        columns = [
            "symbol", "date", "open", "high", "low", "close",
            "volume", "value", "frequency", "foreign_buy", "foreign_sell", "foreign_net"
        ]
        df_all = pd.DataFrame(rows, columns=columns)
        df_all["date"] = pd.to_datetime(df_all["date"])

        numeric_cols = [
            "open", "high", "low", "close", "volume", "value", "frequency",
            "foreign_buy", "foreign_sell", "foreign_net"
        ]
        for col in numeric_cols:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        result = {}
        for symbol, group in df_all.groupby("symbol"):
            stock_df = group.drop("symbol", axis=1).set_index("date").sort_index()
            result[symbol] = stock_df

        return result

    def _score_candidate(
        self,
        dist20: float,
        dist50: float,
        slope50: float,
        reclaim_flag: int
    ) -> float:
        prox_score = self._clamp(1 - (abs(dist20) / RULE_DIST20_ENTRY))
        slope_score = self._clamp(
            (slope50 - RULE_SLOPE_SCORE_FLOOR) /
            (RULE_SLOPE_SCORE_CEIL - RULE_SLOPE_SCORE_FLOOR)
        )
        dist50_score = self._clamp(dist50 / RULE_SCORE_DIST50_CAP)

        total_weight = (
            RULE_SCORE_WEIGHT_PROX
            + RULE_SCORE_WEIGHT_SLOPE
            + RULE_SCORE_WEIGHT_DIST50
            + RULE_SCORE_WEIGHT_RECLAIM
        )
        raw_score = (
            prox_score * RULE_SCORE_WEIGHT_PROX
            + slope_score * RULE_SCORE_WEIGHT_SLOPE
            + dist50_score * RULE_SCORE_WEIGHT_DIST50
            + reclaim_flag * RULE_SCORE_WEIGHT_RECLAIM
        )

        return raw_score / total_weight

    def _latest_signal(self, df: pd.DataFrame) -> Optional[dict]:
        if df.empty or len(df) < (RULE_MA_SLOW + RULE_SLOPE_LOOKBACK + 2):
            return None

        df = df.sort_index()
        return self._signal_for_index(df, len(df) - 1)

    def _signal_for_index(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        if idx < 1:
            return None

        ma20 = df["close"].rolling(window=RULE_MA_FAST).mean()
        ma50 = df["close"].rolling(window=RULE_MA_SLOW).mean()
        slope50 = (ma50 - ma50.shift(RULE_SLOPE_LOOKBACK)) / ma50.shift(RULE_SLOPE_LOOKBACK)

        close = df["close"].iloc[idx]
        low_prev = df["low"].iloc[idx - 1]
        ma20_t = ma20.iloc[idx]
        ma20_prev = ma20.iloc[idx - 1]
        ma50_t = ma50.iloc[idx]
        slope50_t = slope50.iloc[idx]

        if any(pd.isna(x) for x in [ma20_t, ma20_prev, ma50_t, slope50_t]):
            return None

        dist20 = (close - ma20_t) / ma20_t
        dist50 = (close - ma50_t) / ma50_t

        # Hard filters (trend + risk)
        if close <= ma50_t:
            return None
        if slope50_t < RULE_SLOPE_FLAT_MIN:
            return None
        if dist20 > RULE_DIST20_OVEREXTENDED:
            return None
        if dist50 < RULE_DIST50_FALLING_KNIFE:
            return None
        if abs(dist20) > RULE_DIST20_ENTRY:
            return None

        # Trigger: pullback-reclaim OR close-in-zone
        reclaim_flag = int((low_prev <= ma20_prev) and (close > ma20_t))
        close_in_zone = int(close >= ma20_t)
        if reclaim_flag == 0 and close_in_zone == 0:
            return None

        score = self._score_candidate(dist20, dist50, slope50_t, reclaim_flag)

        return {
            "score": score,
            "dist20": dist20,
            "dist50": dist50,
            "slope50": slope50_t,
            "reclaim": reclaim_flag,
            "close_in_zone": close_in_zone,
        }

    def backtest_last_days(self, days: int = 30, top_k: int = None) -> pd.DataFrame:
        top_k = top_k or TOP_PICKS_COUNT

        with session_scope() as session:
            max_date = session.query(DailyPrice.date).order_by(DailyPrice.date.desc()).first()
        if not max_date:
            print("No price data available for backtest.")
            return pd.DataFrame()

        max_date = max_date[0]
        buffer_days = RULE_MA_SLOW + RULE_SLOPE_LOOKBACK + days + 10
        start_date = (max_date - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        end_date = max_date.strftime("%Y-%m-%d")

        symbols = self._get_active_symbols()
        stock_data = self._load_all_stock_data_batch(symbols, start_date=start_date, end_date=end_date)

        with session_scope() as session:
            date_rows = session.query(DailyPrice.date).distinct().order_by(DailyPrice.date).all()
        all_dates = [r[0] for r in date_rows if r and r[0] is not None]
        if len(all_dates) < days:
            print("Not enough dates for backtest.")
            return pd.DataFrame()

        eval_dates = all_dates[-days:]

        results = []
        total_preds = 0
        total_correct = 0

        for eval_date in eval_dates:
            eval_ts = pd.Timestamp(eval_date)
            picks = []
            for symbol, df in stock_data.items():
                if df.empty:
                    continue
                if eval_ts not in df.index:
                    continue

                idx = df.index.get_loc(eval_ts)
                if isinstance(idx, slice):
                    idx = idx.stop - 1
                if idx < 1:
                    continue

                signal = self._signal_for_index(df, idx)
                if signal is None:
                    continue

                if idx + 1 >= len(df):
                    continue

                close_t = df["close"].iloc[idx]
                close_next = df["close"].iloc[idx + 1]
                if close_t == 0 or pd.isna(close_t) or pd.isna(close_next):
                    continue

                ret_next = (close_next - close_t) / close_t
                picks.append({
                    "date": eval_date,
                    "symbol": symbol,
                    "score": signal["score"],
                    "return_next": ret_next,
                    "correct": int(ret_next >= TOP_GAINER_THRESHOLD),
                })

            if not picks:
                results.append({
                    "date": eval_date,
                    "total_picks": 0,
                    "correct": 0,
                    "precision": 0.0,
                })
                continue

            picks_df = pd.DataFrame(picks).sort_values("score", ascending=False).head(top_k)
            correct = int(picks_df["correct"].sum())
            total = int(len(picks_df))
            precision = correct / total if total > 0 else 0.0

            total_preds += total
            total_correct += correct

            results.append({
                "date": eval_date,
                "total_picks": total,
                "correct": correct,
                "precision": precision,
            })

        summary_precision = total_correct / total_preds if total_preds > 0 else 0.0
        print(f"\nBacktest (last {days} days) Precision@{top_k}: {summary_precision:.2%} ({total_correct}/{total_preds})")
        return pd.DataFrame(results)

    def predict(
        self,
        symbols: List[str] = None,
        top_k: int = None,
        save_to_db: bool = True
    ) -> pd.DataFrame:
        top_k = top_k or TOP_PICKS_COUNT
        symbols = symbols or self._get_active_symbols()

        if not symbols:
            print("No symbols found!")
            return pd.DataFrame()

        print("Loading price data...")
        stock_data = self._load_all_stock_data_batch(symbols)

        results = []
        for symbol in symbols:
            df = stock_data.get(symbol)
            if df is None or df.empty:
                continue

            signal = self._latest_signal(df)
            if signal is None:
                continue

            results.append({
                "symbol": symbol,
                "probability": signal["score"],
                "score": signal["score"] * 100,
                "date": df.index[-1],
                "dist20": signal["dist20"],
                "dist50": signal["dist50"],
                "slope50": signal["slope50"],
                "reclaim": signal["reclaim"],
            })

        if not results:
            print("No candidates found.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("probability", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)

        # Optional movers-based filter
        if MOVERS_FILTER_ENABLED:
            mover_date = results_df["date"].iloc[0] if not results_df.empty else None
            if mover_date is not None:
                mover_symbols = self._get_mover_symbols(mover_date, MOVERS_FILTER_TYPES)
                if mover_symbols:
                    results_df = results_df[results_df["symbol"].isin(mover_symbols)].copy()
                    results_df = results_df.sort_values("probability", ascending=False)
                    results_df["rank"] = range(1, len(results_df) + 1)

        if save_to_db:
            self._save_predictions(results_df)

        top_picks = results_df.head(top_k).copy()

        print(f"\nTop {top_k} Ranked Picks:")
        print("-" * 50)
        for _, row in top_picks.iterrows():
            print(f"  {row['rank']:2d}. {row['symbol']:6s} - Score: {row['probability']:.4f}")

        return top_picks

    def _get_mover_symbols(self, date, mover_types: List[str]) -> List[str]:
        from database import DailyMover

        with session_scope() as session:
            rows = session.query(Stock.symbol).join(DailyMover, DailyMover.stock_id == Stock.id).filter(
                DailyMover.date == date,
                DailyMover.mover_type.in_(mover_types)
            ).all()

        return [r[0] for r in rows]

    def _save_predictions(self, results: pd.DataFrame) -> None:
        from utils.holidays import next_trading_day

        prediction_date = datetime.now().date()
        target_date = next_trading_day(prediction_date)

        with session_scope() as session:
            for _, row in results.iterrows():
                stock = session.query(Stock).filter(Stock.symbol == row["symbol"]).first()
                if not stock:
                    continue

                existing = session.query(Prediction).filter(
                    Prediction.stock_id == stock.id,
                    Prediction.prediction_date == prediction_date
                ).first()

                if existing:
                    existing.probability = row["probability"]
                    existing.rank = row["rank"]
                    existing.model_version = self.model_name
                else:
                    pred = Prediction(
                        stock_id=stock.id,
                        prediction_date=prediction_date,
                        target_date=target_date,
                        probability=row["probability"],
                        rank=row["rank"],
                        model_version=self.model_name,
                    )
                    session.add(pred)

    def get_prediction_history(
        self,
        start_date: str = None,
        end_date: str = None,
        top_k: int = None
    ) -> pd.DataFrame:
        top_k = top_k or TOP_PICKS_COUNT

        with session_scope() as session:
            query = session.query(Prediction).join(Stock)

            if start_date:
                query = query.filter(Prediction.prediction_date >= start_date)
            if end_date:
                query = query.filter(Prediction.prediction_date <= end_date)

            query = query.filter(Prediction.rank <= top_k)
            query = query.order_by(Prediction.prediction_date.desc(), Prediction.rank)

            results = []
            for pred in query.all():
                results.append({
                    "prediction_date": pred.prediction_date,
                    "target_date": pred.target_date,
                    "symbol": pred.stock.symbol,
                    "probability": pred.probability,
                    "rank": pred.rank,
                    "actual_return": pred.actual_return,
                    "is_top_gainer": pred.is_top_gainer,
                    "is_correct": pred.is_correct,
                })

            return pd.DataFrame(results)

    def update_actuals(self, date: str = None) -> int:
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date()
        else:
            date = datetime.strptime(date, "%Y-%m-%d").date()

        updated = 0

        with session_scope() as session:
            predictions = session.query(Prediction).filter(
                Prediction.target_date == date,
                Prediction.actual_return == None
            ).all()

            for pred in predictions:
                price_data = session.query(DailyPrice).filter(
                    DailyPrice.stock_id == pred.stock_id,
                    DailyPrice.date == date
                ).first()

                if price_data and price_data.close and price_data.close != 0:
                    prior_price = session.query(DailyPrice).filter(
                        DailyPrice.stock_id == pred.stock_id,
                        DailyPrice.date < date
                    ).order_by(DailyPrice.date.desc()).first()
                    if not prior_price or not prior_price.close:
                        continue

                    pred.actual_return = (price_data.close - prior_price.close) / prior_price.close
                    pred.is_top_gainer = (pred.actual_return >= TOP_GAINER_THRESHOLD)
                    pred.is_correct = (pred.rank is not None and pred.rank <= TOP_PICKS_COUNT) and pred.is_top_gainer
                    updated += 1

        print(f"Updated {updated} predictions with actual returns")
        return updated


def run_prediction():
    """CLI function to run rule-based predictions"""
    import argparse

    parser = argparse.ArgumentParser(description="Run rule-based daily ranking")
    parser.add_argument("--top", type=int, default=TOP_PICKS_COUNT, help="Number of top picks")
    parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    args = parser.parse_args()

    predictor = RuleBasedPredictor()
    results = predictor.predict(top_k=args.top, save_to_db=not args.no_save)

    if not results.empty:
        print("\n" + "=" * 50)
        print("RANKING RESULTS")
        print("=" * 50)
        print(results.to_string(index=False))


if __name__ == "__main__":
    run_prediction()
