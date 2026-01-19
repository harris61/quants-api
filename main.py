"""
Quants-API - Indonesian Stock Market ML Prediction System
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


def cmd_train(args):
    """Train model"""
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer

    init_db()

    print("Building training dataset...")
    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset(
        start_date=args.start,
        end_date=args.end
    )

    if X.empty:
        print("No data available for training!")
        return

    X_train, y_train, feature_names = pipeline.prepare_for_training(X, y)

    print(f"\nTraining model with {len(X_train)} samples...")
    trainer = ModelTrainer(model_name=args.name)
    trainer.train(X_train, y_train, use_smote=not args.no_smote)

    trainer.save()

    print("\nTop 20 Important Features:")
    print(trainer.get_top_features(20))


def cmd_predict(args):
    """Run predictions"""
    from models.predictor import Predictor

    init_db()

    predictor = Predictor(model_name=args.model)
    results = predictor.predict(
        top_k=args.top,
        save_to_db=not args.no_save
    )

    if args.telegram and not results.empty:
        from notifications import TelegramNotifier
        notifier = TelegramNotifier()
        notifier.send_predictions(results)


def cmd_backtest(args):
    """Run backtest"""
    from models.backtester import Backtester

    init_db()

    backtester = Backtester(
        train_days=args.train_days,
        test_days=args.test_days,
        top_k=args.top_k
    )

    results = backtester.run_walk_forward(
        start_date=args.start,
        end_date=args.end
    )

    if args.save and not results.empty:
        backtester.save_results(results)


def cmd_verify(args):
    """Verify data and model"""
    from collectors import HistoricalDataLoader
    from models.predictor import Predictor

    init_db()

    # Data verification
    print("=" * 50)
    print("DATA VERIFICATION")
    print("=" * 50)

    loader = HistoricalDataLoader()
    verification = loader.verify_data()
    for key, value in verification.items():
        print(f"  {key}: {value}")

    # Model verification
    print("\n" + "=" * 50)
    print("MODEL VERIFICATION")
    print("=" * 50)

    try:
        predictor = Predictor()
        print(f"  Model loaded: {predictor.model_name}")
        print(f"  Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"  No model available: {e}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Quants-API - Indonesian Stock Market ML Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py init                      # Initialize database
  python main.py collect-stocks            # Collect stock list
  python main.py load-historical --days 365  # Load 1 year of data
  python main.py train                     # Train model
  python main.py predict                   # Run predictions
  python main.py backtest                  # Run backtest
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

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--start", type=str, help="Training start date")
    train_parser.add_argument("--end", type=str, help="Training end date")
    train_parser.add_argument("--name", type=str, help="Model name")
    train_parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument("--model", type=str, help="Model name to use")
    predict_parser.add_argument("--top", type=int, default=10, help="Number of top picks")
    predict_parser.add_argument("--no-save", action="store_true", help="Don't save to database")
    predict_parser.add_argument("--telegram", action="store_true", help="Send via Telegram")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest_parser.add_argument("--start", type=str, help="Backtest start date")
    backtest_parser.add_argument("--end", type=str, help="Backtest end date")
    backtest_parser.add_argument("--train-days", type=int, default=180, help="Training window days")
    backtest_parser.add_argument("--test-days", type=int, default=30, help="Test window days")
    backtest_parser.add_argument("--top-k", type=int, default=10, help="Top K predictions")
    backtest_parser.add_argument("--save", action="store_true", help="Save results to CSV")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify data and model")

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
        "load-historical": cmd_load_historical,
        "train": cmd_train,
        "predict": cmd_predict,
        "backtest": cmd_backtest,
        "verify": cmd_verify,
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
