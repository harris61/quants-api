"""
Rule-Based Predictor - MA50 + Momentum + Foreign Flow daily ranking (long-only)

Strategy Components:
- MA50 trend filter (must be above MA50, not overextended, slope not falling)
- 5-component scoring: momentum (32%), slope (23%), dist50 (18%), volume (17%), foreign (10%)

Performance: 27.50% precision (2.7x better than random)
"""

from datetime import datetime, timedelta
from typing import List, Optional
import re

import pandas as pd

from config import (
    RULE_BASED_MODEL_NAME,
    RULE_MA_SLOW,
    RULE_SLOPE_LOOKBACK,
    RULE_DIST50_MIN,
    RULE_DIST50_MAX,
    RULE_SLOPE_FLAT_MIN,
    RULE_MOMENTUM_PERIOD,
    RULE_MOMENTUM_FLOOR,
    RULE_MOMENTUM_CEIL,
    RULE_VOLUME_MA_PERIOD,
    RULE_VOLUME_RATIO_FLOOR,
    RULE_VOLUME_RATIO_CEIL,
    RULE_SLOPE_SCORE_FLOOR,
    RULE_SLOPE_SCORE_CEIL,
    RULE_FOREIGN_FLOOR,
    RULE_FOREIGN_CEIL,
    RULE_SCORE_WEIGHT_MOMENTUM,
    RULE_SCORE_WEIGHT_SLOPE,
    RULE_SCORE_WEIGHT_DIST50,
    RULE_SCORE_WEIGHT_VOLUME,
    RULE_SCORE_WEIGHT_FOREIGN,
    RULE_SCORE_DIST50_CAP,
    TOP_PICKS_COUNT,
    TOP_GAINER_THRESHOLD,
    EQUITY_SYMBOL_REGEX,
)
from database import session_scope, Stock, DailyPrice, Prediction, CorporateAction


class RuleBasedPredictor:
    """
    Generate daily ranked picks using MA50 + Momentum + Foreign Flow strategy.

    Filters:
        - Above MA50: close > MA50
        - Not overextended: dist50 < 15%
        - Slope not falling: MA50 slope > -0.2%

    Scoring (weights sum to 100):
        - Momentum (32%): 5-day price return
        - Slope (23%): MA50 trend direction
        - Dist50 (18%): Position above MA50
        - Volume (17%): Volume vs 20-day average
        - Foreign (10%): Net foreign flow (weight is 0 if no data)
    """

    def __init__(self) -> None:
        self.model_name = RULE_BASED_MODEL_NAME
        self.model_type = "ranking"
        self._equity_pattern = re.compile(EQUITY_SYMBOL_REGEX)

    def _clamp(self, value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(value, max_value))

    def _is_equity_symbol(self, symbol: str) -> bool:
        """Filter for normal Indonesian equities only (exclude ETFs, indices, reksadana)."""
        if not symbol:
            return False
        symbol = symbol.upper()

        # Must match 4-letter pattern
        if not self._equity_pattern.match(symbol):
            return False

        # Exclude ETFs (start with X)
        if symbol.startswith("X"):
            return False

        # Exclude known indices and non-equity instruments
        excluded = {
            "FTSE", "KLCI", "IHSG", "ISSI", "LQ45", "IDX3", "IDXG", "IDXV",
            "IDXQ", "IDXE", "IDXS", "IDXH", "IDXI", "IDXC", "IDXF", "IDXB",
            "PEFINDO25", "SRI-KEHATI",
        }
        if symbol in excluded:
            return False

        return True

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

        actions_map = self._load_corporate_actions(symbols, start_date=start_date, end_date=end_date)

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
            if symbol in actions_map:
                stock_df = self._apply_corporate_actions(stock_df, actions_map[symbol])
            result[symbol] = self._precompute_indicators(stock_df)

        return result

    def _precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute MA50 and derived indicators for efficiency."""
        df = df.copy()
        df["ma50"] = df["close"].rolling(window=RULE_MA_SLOW).mean()
        df["slope50"] = (df["ma50"] - df["ma50"].shift(RULE_SLOPE_LOOKBACK)) / df["ma50"].shift(RULE_SLOPE_LOOKBACK)
        df["dist50"] = (df["close"] - df["ma50"]) / df["ma50"]
        df["momentum"] = df["close"].pct_change(periods=RULE_MOMENTUM_PERIOD)
        df["volume_ma"] = df["volume"].rolling(window=RULE_VOLUME_MA_PERIOD).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        return df

    def _load_corporate_actions(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> dict:
        if not symbols:
            return {}

        def _to_date(value):
            if value is None or not isinstance(value, str):
                return value
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                return value

        start_date = _to_date(start_date)
        end_date = _to_date(end_date)

        with session_scope() as session:
            query = session.query(CorporateAction).filter(
                CorporateAction.symbol.in_([s.upper() for s in symbols]),
                CorporateAction.action_type.in_(["stocksplit", "dividend"]),
                CorporateAction.ex_date != None,
                CorporateAction.value != None,
            )
            if start_date:
                query = query.filter(CorporateAction.ex_date >= start_date)
            if end_date:
                query = query.filter(CorporateAction.ex_date <= end_date)

            actions = query.all()

        actions_map = {}
        for action in actions:
            symbol = action.symbol
            actions_map.setdefault(symbol, []).append(action)

        return actions_map

    def _apply_corporate_actions(self, df: pd.DataFrame, actions: list) -> pd.DataFrame:
        """Apply backward adjustments for stock splits and cash dividends."""
        if df.empty or not actions:
            return df

        df = df.copy()
        price_cols = ["open", "high", "low", "close"]

        for action in sorted(actions, key=lambda x: x.ex_date or datetime.min.date()):
            ex_date = pd.Timestamp(action.ex_date)
            if ex_date not in df.index and not (df.index < ex_date).any():
                continue

            action_type = (action.action_type or "").lower()
            if action_type == "stocksplit":
                ratio = action.value
                if ratio is None:
                    continue
                try:
                    ratio = float(ratio)
                except (TypeError, ValueError):
                    continue
                if ratio <= 0 or ratio == 1:
                    continue
                mask = df.index < ex_date
                if not mask.any():
                    continue
                df.loc[mask, price_cols] = df.loc[mask, price_cols] / ratio
                df.loc[mask, "volume"] = df.loc[mask, "volume"] * ratio
                continue

            if action_type == "dividend":
                cash_div = action.value
                if cash_div is None:
                    continue
                try:
                    cash_div = float(cash_div)
                except (TypeError, ValueError):
                    continue
                if cash_div <= 0:
                    continue
                mask = df.index < ex_date
                if not mask.any():
                    continue
                # Approximate cash dividend back-adjustment; floor to avoid negatives.
                df.loc[mask, price_cols] = (df.loc[mask, price_cols] - cash_div).clip(lower=0.01)

        return df

    def _score_candidate(
        self,
        momentum: float,
        slope50: float,
        dist50: float,
        volume_ratio: float,
        foreign_net: float = None
    ) -> float:
        return self._score_components(
            momentum, slope50, dist50, volume_ratio, foreign_net
        )["score"]

    def _score_components(
        self,
        momentum: float,
        slope50: float,
        dist50: float,
        volume_ratio: float,
        foreign_net: float = None
    ) -> dict:
        momentum_score = self._clamp(
            (momentum - RULE_MOMENTUM_FLOOR) /
            (RULE_MOMENTUM_CEIL - RULE_MOMENTUM_FLOOR)
        )
        slope_score = self._clamp(
            (slope50 - RULE_SLOPE_SCORE_FLOOR) /
            (RULE_SLOPE_SCORE_CEIL - RULE_SLOPE_SCORE_FLOOR)
        )
        dist50_score = self._clamp(dist50 / RULE_SCORE_DIST50_CAP)
        volume_score = self._clamp(
            (volume_ratio - RULE_VOLUME_RATIO_FLOOR) /
            (RULE_VOLUME_RATIO_CEIL - RULE_VOLUME_RATIO_FLOOR)
        )

        # Foreign flow score (if available)
        if foreign_net is not None and not pd.isna(foreign_net):
            foreign_score = self._clamp(
                (foreign_net - RULE_FOREIGN_FLOOR) /
                (RULE_FOREIGN_CEIL - RULE_FOREIGN_FLOOR)
            )
            foreign_weight = RULE_SCORE_WEIGHT_FOREIGN
        else:
            foreign_score = 0.0
            foreign_weight = 0

        total_weight = (
            RULE_SCORE_WEIGHT_MOMENTUM
            + RULE_SCORE_WEIGHT_SLOPE
            + RULE_SCORE_WEIGHT_DIST50
            + RULE_SCORE_WEIGHT_VOLUME
            + foreign_weight
        )
        raw_score = (
            momentum_score * RULE_SCORE_WEIGHT_MOMENTUM
            + slope_score * RULE_SCORE_WEIGHT_SLOPE
            + dist50_score * RULE_SCORE_WEIGHT_DIST50
            + volume_score * RULE_SCORE_WEIGHT_VOLUME
            + foreign_score * foreign_weight
        )

        return {
            "score": raw_score / total_weight,
            "momentum_score": momentum_score,
            "slope_score": slope_score,
            "dist50_score": dist50_score,
            "volume_score": volume_score,
            "foreign_score": foreign_score,
        }

    def _latest_signal(self, df: pd.DataFrame) -> Optional[dict]:
        min_required = max(RULE_MA_SLOW, RULE_MOMENTUM_PERIOD, RULE_VOLUME_MA_PERIOD) + RULE_SLOPE_LOOKBACK + 2
        if df.empty or len(df) < min_required:
            return None

        df = df.sort_index()
        return self._signal_for_index(df, len(df) - 1)

    def _signal_for_index(self, df: pd.DataFrame, idx: int) -> Optional[dict]:
        if idx < 1:
            return None

        # Use pre-computed indicators if available, otherwise compute on-the-fly
        if "ma50" not in df.columns:
            df = self._precompute_indicators(df)

        close = df["close"].iloc[idx]
        ma50_t = df["ma50"].iloc[idx]
        slope50_t = df["slope50"].iloc[idx]
        dist50 = df["dist50"].iloc[idx]
        momentum = df["momentum"].iloc[idx]
        volume_ma_t = df["volume_ma"].iloc[idx]
        volume_ratio = df["volume_ratio"].iloc[idx]
        foreign_net = df["foreign_net"].iloc[idx] if "foreign_net" in df.columns else None

        if any(pd.isna(x) for x in [ma50_t, slope50_t, dist50, momentum, volume_ma_t, volume_ratio]):
            return None
        if close <= 0 or volume_ma_t <= 0:
            return None

        # Hard filters (MA50-based only)
        # 1. Must be above MA50
        if dist50 < RULE_DIST50_MIN:
            return None
        # 2. Not too extended above MA50
        if dist50 > RULE_DIST50_MAX:
            return None
        # 3. MA50 slope must not be falling sharply
        if slope50_t < RULE_SLOPE_FLAT_MIN:
            return None

        components = self._score_components(momentum, slope50_t, dist50, volume_ratio, foreign_net)
        score = components["score"]

        return {
            "score": score,
            "momentum": momentum,
            "dist50": dist50,
            "slope50": slope50_t,
            "volume_ratio": volume_ratio,
            "foreign_net": foreign_net,
            "momentum_score": components["momentum_score"],
            "slope_score": components["slope_score"],
            "dist50_score": components["dist50_score"],
            "volume_score": components["volume_score"],
            "foreign_score": components["foreign_score"],
        }

    def backtest_last_days(self, days: int = 30, top_k: int = None, save_csv: bool = False) -> pd.DataFrame:
        from utils.holidays import is_trading_day
        from config import BASE_DIR

        top_k = top_k or TOP_PICKS_COUNT
        top_k = min(top_k, TOP_PICKS_COUNT)

        with session_scope() as session:
            max_date = session.query(DailyPrice.date).order_by(DailyPrice.date.desc()).first()
        if not max_date:
            print("No price data available for backtest.")
            return pd.DataFrame()

        max_date = max_date[0]
        # Buffer needs to account for: MA50 warmup + slope lookback + test days + margin
        # Multiply by 1.5 to convert trading days to calendar days (weekends/holidays)
        buffer_days = int((RULE_MA_SLOW + RULE_SLOPE_LOOKBACK + days) * 1.5) + 20
        start_date = (max_date - timedelta(days=buffer_days)).strftime("%Y-%m-%d")
        end_date = max_date.strftime("%Y-%m-%d")

        symbols = self._get_active_symbols()
        stock_data = self._load_all_stock_data_batch(symbols, start_date=start_date, end_date=end_date)

        with session_scope() as session:
            date_rows = session.query(DailyPrice.date).distinct().order_by(DailyPrice.date).all()
        # Filter to trading days only (exclude weekends)
        all_dates = [r[0] for r in date_rows if r and r[0] is not None and is_trading_day(r[0])]
        if len(all_dates) < days:
            print("Not enough trading dates for backtest.")
            return pd.DataFrame()

        eval_dates = all_dates[-days:]

        results = []
        all_picks = []  # Collect all picks for CSV export
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
                    "momentum": signal["momentum"],
                    "dist50": signal["dist50"],
                    "slope50": signal["slope50"],
                    "volume_ratio": signal["volume_ratio"],
                    "close": close_t,
                    "close_next": close_next,
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

            picks_df = pd.DataFrame(picks).sort_values("score", ascending=False)
            # Add rank column before filtering to top_k
            picks_df["rank"] = range(1, len(picks_df) + 1)
            top_picks_df = picks_df.head(top_k).copy()

            # Add to all_picks for CSV export
            all_picks.extend(top_picks_df.to_dict("records"))

            correct = int(top_picks_df["correct"].sum())
            total = int(len(top_picks_df))
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
        print(f"\nBacktest ({len(eval_dates)} trading days) Precision@{top_k}: {summary_precision:.2%} ({total_correct}/{total_preds})")

        # Build full results DataFrame with stockpick columns
        picks_df = pd.DataFrame(all_picks) if all_picks else pd.DataFrame()
        summary_df = pd.DataFrame(results)

        # Add stockpick columns to summary
        for i in range(1, top_k + 1):
            summary_df[f"stockpick_{i}"] = ""

        for idx, row in summary_df.iterrows():
            eval_date = row["date"]
            if not picks_df.empty:
                date_picks = picks_df[picks_df["date"] == eval_date].sort_values("rank")
                for _, pick in date_picks.iterrows():
                    rank = int(pick["rank"])
                    ret_pct = pick["return_next"] * 100
                    summary_df.at[idx, f"stockpick_{rank}"] = f"{pick['symbol']}/{ret_pct:+.1f}%"

        # Save to CSV
        if save_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = BASE_DIR / f"backtest_{timestamp}.csv"
            summary_df.to_csv(csv_path, index=False)
            print(f"Backtest results saved to: {csv_path}")

        return summary_df

    def predict(
        self,
        symbols: List[str] = None,
        top_k: int = None,
        save_to_db: bool = True,
        include_components: bool = False
    ) -> pd.DataFrame:
        top_k = top_k or TOP_PICKS_COUNT
        top_k = min(top_k, TOP_PICKS_COUNT)
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

            row = {
                "symbol": symbol,
                "probability": signal["score"],
                "score": signal["score"] * 100,
                "date": df.index[-1],
                "momentum": signal["momentum"],
                "dist50": signal["dist50"],
                "slope50": signal["slope50"],
                "volume_ratio": signal["volume_ratio"],
            }
            if include_components:
                row.update({
                    "momentum_score": signal.get("momentum_score"),
                    "slope_score": signal.get("slope_score"),
                    "dist50_score": signal.get("dist50_score"),
                    "volume_score": signal.get("volume_score"),
                    "foreign_score": signal.get("foreign_score"),
                })

            results.append(row)

        if not results:
            print("No candidates found.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("probability", ascending=False)
        results_df["rank"] = range(1, len(results_df) + 1)

        if save_to_db:
            self._save_predictions(results_df)

        top_picks = results_df.head(top_k).copy()

        # Get prediction date from the data
        pred_date = top_picks["date"].iloc[0] if not top_picks.empty else datetime.now().date()
        if hasattr(pred_date, 'strftime'):
            pred_date_str = pred_date.strftime("%Y-%m-%d")
        else:
            pred_date_str = str(pred_date)

        print(f"\nTop {top_k} Ranked Picks (Data: {pred_date_str})")
        print("-" * 70)
        name_map = {}
        with session_scope() as session:
            rows = session.query(Stock.symbol, Stock.name).filter(
                Stock.symbol.in_(top_picks["symbol"].tolist())
            ).all()
        name_map = {symbol: name or "" for symbol, name in rows}
        for _, row in top_picks.iterrows():
            name = name_map.get(row["symbol"], "")
            label = f"{row['symbol']:6s} {name}" if name else f"{row['symbol']:6s}"
            print(f"  {row['rank']:2d}. {label} - Score: {row['probability']:.4f}")

        return top_picks

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
