"""
Daily Run Script - Automated daily workflow
Run this script daily after market close (after 4:30 PM WIB)
"""

import logging
from datetime import datetime, time
import schedule
import time as time_module
from typing import Optional

from database import init_db
from collectors import (
    DailyDataCollector,
    HistoricalDataLoader,
    BrokerSummaryCollector,
    InsiderTradeCollector,
    IntradayCollector,
)
from models.predictor import Predictor
from notifications import TelegramNotifier
from config import LOG_FILE, LOG_LEVEL, TOP_PICKS_COUNT

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_trading_day() -> bool:
    """Check if today is a trading day (Mon-Fri)"""
    return datetime.now().weekday() < 5


def run_daily_workflow(send_telegram: bool = True) -> dict:
    """
    Execute the complete daily workflow

    Steps:
    1. Collect today's data
    2. Update yesterday's prediction results
    3. Run predictions for tomorrow
    4. Send notifications

    Args:
        send_telegram: Whether to send Telegram notifications

    Returns:
        Dict with workflow results
    """
    results = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "started",
        "data_collected": False,
        "predictions_made": False,
        "notification_sent": False,
        "errors": []
    }

    logger.info("=" * 60)
    logger.info(f"Starting daily workflow - {results['date']}")
    logger.info("=" * 60)

    # Initialize database
    init_db()

    # Step 1: Collect today's data
    logger.info("\n[Step 1] Collecting today's market data...")
    try:
        collector = DailyDataCollector()
        stats = collector.collect_today()
        results["data_collected"] = stats["success"] > 0
        results["data_stats"] = stats
        logger.info(f"Data collection: {stats['success']} stocks, {stats['records']} records")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        results["errors"].append(f"Data collection: {e}")

    # Step 1b: Collect broker summary data
    logger.info("\n[Step 1b] Collecting broker summary data...")
    try:
        broker_collector = BrokerSummaryCollector()
        broker_stats = broker_collector.collect_today()
        results["broker_collected"] = broker_stats["success"] > 0
        results["broker_stats"] = broker_stats
        logger.info(f"Broker collection: {broker_stats['success']} stocks, {broker_stats['records']} records")
    except Exception as e:
        logger.error(f"Broker collection failed: {e}")
        results["errors"].append(f"Broker collection: {e}")

    # Step 1c: Collect intraday data
    logger.info("\n[Step 1c] Collecting intraday data...")
    try:
        intraday_collector = IntradayCollector()
        intraday_stats = intraday_collector.collect_today()
        results["intraday_collected"] = intraday_stats["success"] > 0
        results["intraday_stats"] = intraday_stats
        logger.info(f"Intraday collection: {intraday_stats['success']} stocks, {intraday_stats['records']} records")
    except Exception as e:
        logger.error(f"Intraday collection failed: {e}")
        results["errors"].append(f"Intraday collection: {e}")

    # Step 1d: Collect insider data (weekly - Monday only)
    if datetime.now().weekday() == 0:  # Monday
        logger.info("\n[Step 1d] Collecting insider trading data (weekly)...")
        try:
            insider_collector = InsiderTradeCollector()
            insider_stats = insider_collector.collect_and_save()
            results["insider_collected"] = insider_stats["success"] > 0
            results["insider_stats"] = insider_stats
            logger.info(f"Insider collection: {insider_stats['success']} stocks, {insider_stats['new_records']} new records")
        except Exception as e:
            logger.error(f"Insider collection failed: {e}")
            results["errors"].append(f"Insider collection: {e}")

    # Step 2: Update yesterday's prediction results
    logger.info("\n[Step 2] Updating prediction results...")
    try:
        predictor = Predictor()
        updated = predictor.update_actuals()
        results["actuals_updated"] = updated
        logger.info(f"Updated {updated} prediction results")
    except Exception as e:
        logger.error(f"Failed to update actuals: {e}")
        results["errors"].append(f"Update actuals: {e}")

    # Step 3: Run predictions
    logger.info("\n[Step 3] Running predictions...")
    try:
        predictor = Predictor()
        predictions = predictor.predict(top_k=TOP_PICKS_COUNT, save_to_db=True)
        results["predictions_made"] = not predictions.empty
        results["predictions"] = predictions.to_dict('records') if not predictions.empty else []
        logger.info(f"Generated {len(predictions)} predictions")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        results["errors"].append(f"Prediction: {e}")
        predictions = None

    # Step 4: Send Telegram notification
    if send_telegram and predictions is not None and not predictions.empty:
        logger.info("\n[Step 4] Sending Telegram notification...")
        try:
            notifier = TelegramNotifier()

            # Get yesterday's results for summary
            history = predictor.get_prediction_history()
            yesterday_results = None
            if not history.empty and len(history) > TOP_PICKS_COUNT:
                yesterday_results = history.tail(TOP_PICKS_COUNT)

            success = notifier.send_daily_summary(predictions, yesterday_results)
            results["notification_sent"] = success
            logger.info(f"Telegram notification: {'sent' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            results["errors"].append(f"Telegram: {e}")

    # Final status
    if results["errors"]:
        results["status"] = "completed_with_errors"
    else:
        results["status"] = "completed"

    logger.info("\n" + "=" * 60)
    logger.info(f"Daily workflow completed - Status: {results['status']}")
    logger.info("=" * 60)

    return results


def run_scheduled():
    """Run with scheduler - executes daily at specified time"""

    # Schedule for after market close (4:45 PM WIB / 09:45 UTC)
    schedule_time = "16:45"

    logger.info(f"Scheduler started. Will run daily at {schedule_time}")
    logger.info("Press Ctrl+C to stop.")

    schedule.every().day.at(schedule_time).do(run_daily_workflow)

    # Also run immediately if it's past schedule time on a trading day
    now = datetime.now()
    schedule_dt = datetime.strptime(schedule_time, "%H:%M").replace(
        year=now.year, month=now.month, day=now.day
    )

    if is_trading_day() and now > schedule_dt:
        logger.info("Past scheduled time today, running now...")
        run_daily_workflow()

    while True:
        schedule.run_pending()
        time_module.sleep(60)


def run_initial_setup():
    """
    Run initial setup for new installation

    Steps:
    1. Initialize database
    2. Collect stock list
    3. Load historical data
    4. Train initial model
    """
    from collectors import StockListCollector, HistoricalDataLoader
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer

    logger.info("=" * 60)
    logger.info("INITIAL SETUP")
    logger.info("=" * 60)

    # Step 1: Initialize database
    logger.info("\n[Step 1] Initializing database...")
    init_db()

    # Step 2: Collect stock list
    logger.info("\n[Step 2] Collecting stock list...")
    stock_collector = StockListCollector()
    stocks = stock_collector.collect_all_stocks()
    stock_collector.save_to_database(stocks)

    # Step 3: Load historical data
    logger.info("\n[Step 3] Loading historical data (this may take a while)...")
    loader = HistoricalDataLoader(days=365)
    loader.load_historical_data()

    # Step 4: Train initial model
    logger.info("\n[Step 4] Training initial model...")
    pipeline = FeaturePipeline()
    X, y = pipeline.build_training_dataset()

    if not X.empty:
        X_train, y_train, _ = pipeline.prepare_for_training(X, y)
        trainer = ModelTrainer(model_name="initial_model")
        trainer.train(X_train, y_train)
        trainer.save()
        logger.info("Model trained and saved!")
    else:
        logger.warning("Not enough data to train model!")

    logger.info("\n" + "=" * 60)
    logger.info("Initial setup complete!")
    logger.info("You can now run: python daily_run.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quants-API Daily Runner")
    parser.add_argument("--scheduled", action="store_true",
                        help="Run with scheduler (continuous)")
    parser.add_argument("--setup", action="store_true",
                        help="Run initial setup")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Skip Telegram notification")
    args = parser.parse_args()

    if args.setup:
        run_initial_setup()
    elif args.scheduled:
        run_scheduled()
    else:
        run_daily_workflow(send_telegram=not args.no_telegram)
