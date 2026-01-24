"""
Quants-API - Indonesian Stock Market Rule-Based Ranking System
Main CLI Entry Point
"""

import argparse
import sys
from datetime import datetime

from database import init_db


def cmd_init(args):
    """Initialize database"""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")


def cmd_collect_stocks(args):
    """Collect stock list"""
    from collectors import StockListCollector

    init_db()
    collector = StockListCollector()
    stocks = collector.collect_all_stocks()
    collector.save_to_database(stocks)


def cmd_collect_data(args):
    """Collect daily data"""
    from collectors import DailyDataCollector

    init_db()
    collector = DailyDataCollector()

    if args.days:
        collector.collect_and_save(days=args.days)
    else:
        collector.collect_today()


def cmd_load_historical(args):
    """Load historical data"""
    from collectors import HistoricalDataLoader

    init_db()
    loader = HistoricalDataLoader(days=args.days)

    if args.verify:
        verification = loader.verify_data()
        print("\nData Verification:")
        for key, value in verification.items():
            print(f"  {key}: {value}")
    else:
        loader.load_historical_data(resume=not args.no_resume)


def cmd_predict(args):
    """Run predictions"""
    from models.rule_based import RuleBasedPredictor

    init_db()

    predictor = RuleBasedPredictor()
    top_k = min(args.top, 5)
    results = predictor.predict(
        top_k=top_k,
        save_to_db=not args.no_save
    )

    if args.telegram and not results.empty:
        from notifications import TelegramNotifier
        notifier = TelegramNotifier()
        notifier.send_predictions(results)


def cmd_predict_scores(args):
    """Run predictions with component scores for review"""
    from models.rule_based import RuleBasedPredictor

    init_db()

    predictor = RuleBasedPredictor()
    top_k = min(args.top, 5)
    results = predictor.predict(
        top_k=top_k,
        save_to_db=False,
        include_components=True
    )

    if not results.empty:
        print("\n" + "=" * 50)
        print("PREDICTION COMPONENT SCORES")
        print("=" * 50)
        print(results.to_string(index=False))


def cmd_backtest_rules(args):
    """Run rule-based backtest"""
    from models.rule_based import RuleBasedPredictor

    init_db()

    predictor = RuleBasedPredictor()
    save_csv = bool(getattr(args, 'csv', False))
    if getattr(args, 'no_csv', False):
        save_csv = False
    top_k = min(args.top, 5)
    results = predictor.backtest_last_days(days=args.days, top_k=top_k, save_csv=save_csv)

    if not results.empty:
        print("\n" + "=" * 50)
        print("RULE-BASED BACKTEST")
        print("=" * 50)
        print(results.to_string(index=False))


def cmd_verify(args):
    """Verify data and rule-based system"""
    from collectors import HistoricalDataLoader
    from models.rule_based import RuleBasedPredictor

    init_db()

    # Data verification
    print("=" * 50)
    print("DATA VERIFICATION")
    print("=" * 50)

    loader = HistoricalDataLoader()
    verification = loader.verify_data()
    for key, value in verification.items():
        print(f"  {key}: {value}")

    # Rule-based verification
    print("\n" + "=" * 50)
    print("RULE-BASED VERIFICATION")
    print("=" * 50)

    try:
        predictor = RuleBasedPredictor()
        print(f"  Strategy: {predictor.model_name}")
    except Exception as e:
        print(f"  Rule-based system error: {e}")


def cmd_verify_range(args):
    """Verify data coverage for a date range"""
    from database import session_scope, DailyPrice

    start_date = args.start
    end_date = args.end

    with session_scope() as session:
        query = session.query(DailyPrice)
        if start_date:
            query = query.filter(DailyPrice.date >= start_date)
        if end_date:
            query = query.filter(DailyPrice.date <= end_date)

        total_records = query.count()
        distinct_stocks = query.with_entities(DailyPrice.stock_id).distinct().count()

        min_date = query.with_entities(DailyPrice.date).order_by(DailyPrice.date.asc()).first()
        max_date = query.with_entities(DailyPrice.date).order_by(DailyPrice.date.desc()).first()

    print("RANGE VERIFICATION")
    print(f"  start: {start_date or 'N/A'}")
    print(f"  end: {end_date or 'N/A'}")
    print(f"  total_records: {total_records}")
    print(f"  distinct_stocks: {distinct_stocks}")
    print(f"  min_date: {min_date[0] if min_date else None}")
    print(f"  max_date: {max_date[0] if max_date else None}")


def cmd_telegram_test(args):
    """Test Telegram notification"""
    from notifications import TelegramNotifier

    notifier = TelegramNotifier()
    success = notifier.send_alert(
        "info",
        "This is a test message from Quants-API."
    )

    if success:
        print("Telegram test message sent successfully!")
    else:
        print("Failed to send Telegram message. Check your configuration.")


def cmd_daily_run(args):
    """Run complete daily workflow"""
    from daily_run import run_daily_workflow
    run_daily_workflow(send_telegram=not args.no_telegram)


def cmd_collect_broker(args):
    """Collect broker summary data"""
    from collectors import BrokerSummaryCollector

    init_db()
    collector = BrokerSummaryCollector()
    collector.collect_today()


def cmd_collect_insider(args):
    """Collect insider trading data"""
    from collectors import InsiderTradeCollector

    init_db()
    collector = InsiderTradeCollector()
    collector.collect_and_save()


def cmd_collect_intraday(args):
    """Collect intraday OHLCV data"""
    from collectors import IntradayCollector

    init_db()
    collector = IntradayCollector()
    collector.collect_and_save(days=args.days)


def cmd_collect_foreign(args):
    """Collect foreign flow data from net foreign endpoints"""
    from collectors import DailyDataCollector
    from datetime import datetime, timedelta

    init_db()
    collector = DailyDataCollector()

    if getattr(args, "fill_missing", False):
        if not args.start or not args.end:
            print("Missing --start/--end for fill-missing mode.")
            return
        collector.backfill_foreign_flow_missing(
            start_date=args.start,
            end_date=args.end
        )
    elif args.backfill:
        # Backfill historical data
        if not args.start or not args.end:
            # Default to last 30 days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        else:
            start_date = args.start
            end_date = args.end
        print(f"Backfilling foreign flow from {start_date} to {end_date}")
        print("WARNING: This is slow (~0.5s per stock). Press Ctrl+C to cancel.")
        collector.backfill_foreign_flow(start_date=start_date, end_date=end_date)
    else:
        # Collect today's data from net foreign endpoints
        collector.collect_foreign_flow(date=args.date)


def main():
    parser = argparse.ArgumentParser(
        description="Quants-API - Indonesian Stock Market Rule-Based Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py init                      # Initialize database
  python main.py collect-stocks            # Collect stock list
  python main.py load-historical --days 365  # Load 1 year of data
  python main.py predict                   # Run daily ranked picks
  python main.py backtest-rules --days 30  # Run backtest
  python main.py daily                     # Run daily workflow
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")

    # Collect stocks command
    stocks_parser = subparsers.add_parser("collect-stocks", help="Collect stock list from API")

    # Collect data command
    data_parser = subparsers.add_parser("collect-data", help="Collect daily trading data")
    data_parser.add_argument("--days", type=int, help="Number of days to collect")

    # Load historical command
    historical_parser = subparsers.add_parser("load-historical", help="Load historical data")
    historical_parser.add_argument("--days", type=int, default=365, help="Days of history")
    historical_parser.add_argument("--no-resume", action="store_true", help="Don't resume from last date")
    historical_parser.add_argument("--verify", action="store_true", help="Only verify existing data")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run daily ranked picks")
    predict_parser.add_argument("--top", type=int, default=5, help="Number of top picks (max 5)")
    predict_parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    predict_parser.add_argument("--telegram", action="store_true", help="Send via Telegram")

    # Predict with component scores
    predict_scores_parser = subparsers.add_parser("predict-scores", help="Run predictions with component scores")
    predict_scores_parser.add_argument("--top", type=int, default=5, help="Number of top picks (max 5)")

    # Backtest command
    rb_backtest_parser = subparsers.add_parser("backtest-rules", help="Run rule-based backtest")
    rb_backtest_parser.add_argument("--days", type=int, default=30, help="Number of recent days to test")
    rb_backtest_parser.add_argument("--top", type=int, default=5, help="Top K picks per day (max 5)")
    rb_backtest_parser.add_argument("--csv", action="store_true", help="Save CSV files")
    rb_backtest_parser.add_argument("--no-csv", action="store_true", help="Don't save CSV files (deprecated)")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify data and rule-based system")

    # Verify date range command
    verify_range_parser = subparsers.add_parser("verify-range", help="Verify data coverage for a date range")
    verify_range_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    verify_range_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    # Telegram test command
    telegram_parser = subparsers.add_parser("telegram-test", help="Test Telegram notification")

    # Daily run command
    daily_parser = subparsers.add_parser("daily", help="Run complete daily workflow")
    daily_parser.add_argument("--no-telegram", action="store_true", help="Skip Telegram notification")

    # Collect broker data command
    broker_parser = subparsers.add_parser("collect-broker", help="Collect broker summary data")

    # Collect insider data command
    insider_parser = subparsers.add_parser("collect-insider", help="Collect insider trading data")

    # Collect intraday data command
    intraday_parser = subparsers.add_parser("collect-intraday", help="Collect intraday OHLCV data")
    intraday_parser.add_argument("--days", type=int, default=5, help="Days of intraday data to collect")

    # Collect foreign flow data command
    foreign_parser = subparsers.add_parser("collect-foreign", help="Collect foreign flow data")
    foreign_parser.add_argument("--date", type=str, help="Date to collect (YYYY-MM-DD), defaults to today")
    foreign_parser.add_argument("--backfill", action="store_true", help="Backfill historical data (slow)")
    foreign_parser.add_argument("--fill-missing", action="store_true", help="Fill missing foreign flow days (per-day calls)")
    foreign_parser.add_argument("--start", type=str, help="Start date for backfill (YYYY-MM-DD)")
    foreign_parser.add_argument("--end", type=str, help="End date for backfill (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Route to command handler
    commands = {
        "init": cmd_init,
        "collect-stocks": cmd_collect_stocks,
        "collect-data": cmd_collect_data,
        "collect-broker": cmd_collect_broker,
        "collect-insider": cmd_collect_insider,
        "collect-intraday": cmd_collect_intraday,
        "collect-foreign": cmd_collect_foreign,
        "load-historical": cmd_load_historical,
        "predict": cmd_predict,
        "predict-scores": cmd_predict_scores,
        "backtest-rules": cmd_backtest_rules,
        "verify": cmd_verify,
        "verify-range": cmd_verify_range,
        "telegram-test": cmd_telegram_test,
        "daily": cmd_daily_run,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
