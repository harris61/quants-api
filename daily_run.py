"""
Daily Run Script - Automated daily workflow
Run this script daily after market close (after 4:30 PM WIB)
"""

import logging
from datetime import datetime, time
import schedule
import time as time_module
from typing import Optional

from database import init_db, session_scope
from database.models import DailyPrice, Stock
from collectors import (
    DailyDataCollector,
    HistoricalDataLoader,
    BrokerSummaryCollector,
    OrderBookCollector,
)
from models.rule_based import RuleBasedPredictor
from notifications import TelegramNotifier
from config import (
    LOG_FILE,
    LOG_LEVEL,
    TOP_PICKS_COUNT,
    DAILY_COLLECT_DAYS,
    BROKER_COLLECTION_ENABLED,
    DIVERGENCE_ENABLED,
    DIVERGENCE_TOP_N,
    DIVERGENCE_SMART_BROKERS,
    DIVERGENCE_RETAIL_BROKERS,
    ORDERBOOK_COLLECTION_ENABLED,
    ORDERBOOK_SCHEDULE_ENABLED,
    ORDERBOOK_INTERVAL_MINUTES,
    ORDERBOOK_START_TIME,
    ORDERBOOK_END_TIME,
)

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
    """Check if today is a trading day on IDX (excludes weekends and holidays)"""
    from utils.holidays import is_trading_day as check_trading_day
    return check_trading_day(datetime.now().date())


def _get_latest_price_snapshot() -> tuple:
    """Return latest daily_prices date, record count, and active stock count."""
    with session_scope() as session:
        latest_row = session.query(DailyPrice.date).order_by(DailyPrice.date.desc()).first()
        if not latest_row:
            return None, 0, 0
        latest_date = latest_row[0]
        records = session.query(DailyPrice).filter(DailyPrice.date == latest_date).count()
        active = session.query(Stock).filter(Stock.is_active == True).count()
    return latest_date, records, active


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

    if not is_trading_day():
        results["status"] = "skipped_non_trading_day"
        logger.info("Not a trading day. Skipping workflow.")
        return results

    # Initialize database
    init_db()
    collector = None

    # Step 1: Collect today's data
    logger.info("\n[Step 1] Collecting today's market data...")
    try:
        collector = DailyDataCollector()
        stats = collector.collect_and_save(days=DAILY_COLLECT_DAYS)
        results["data_collected"] = stats["success"] > 0
        results["data_stats"] = stats
        logger.info(f"Data collection: {stats['success']} stocks, {stats['records']} records")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        results["errors"].append(f"Data collection: {e}")

    # Step 1b: Collect foreign flow data
    logger.info("\n[Step 1b] Collecting foreign flow data...")
    try:
        if collector is None:
            collector = DailyDataCollector()
        foreign_stats = collector.collect_foreign_flow()
        results["foreign_flow_stats"] = foreign_stats
        logger.info(f"Foreign flow: {foreign_stats['updated']} stocks updated")
    except Exception as e:
        logger.error(f"Foreign flow collection failed: {e}")
        results["errors"].append(f"Foreign flow: {e}")

    # Step 1c: Collect broker summary data
    if BROKER_COLLECTION_ENABLED:
        logger.info("\n[Step 1c] Collecting broker summary data...")
        try:
            broker_collector = BrokerSummaryCollector()
            broker_stats = broker_collector.collect_today()
            results["broker_summary_stats"] = broker_stats
            logger.info(
                f"Broker summary: {broker_stats['success']} stocks, {broker_stats['records']} records"
            )
        except Exception as e:
            logger.error(f"Broker summary collection failed: {e}")
            results["errors"].append(f"Broker summary: {e}")

    # Step 1c2: Collect orderbook snapshots
    if ORDERBOOK_COLLECTION_ENABLED:
        logger.info("\n[Step 1c2] Collecting orderbook snapshots...")
        try:
            orderbook_collector = OrderBookCollector()
            orderbook_stats = orderbook_collector.collect_and_save(show_progress=False)
            results["orderbook_stats"] = orderbook_stats
            logger.info(
                f"Orderbook: {orderbook_stats['success']} stocks, "
                f"{orderbook_stats['records']} records"
            )
        except Exception as e:
            logger.error(f"Orderbook collection failed: {e}")
            results["errors"].append(f"Orderbook: {e}")

    # Step 1d: Divergence analysis
    if DIVERGENCE_ENABLED:
        logger.info("\n[Step 1d] Running divergence analysis...")
        try:
            from scripts.divergence_analysis import run as run_divergence
            today = datetime.now().strftime("%Y-%m-%d")
            run_divergence(
                today,
                DIVERGENCE_SMART_BROKERS,
                DIVERGENCE_RETAIL_BROKERS,
                DIVERGENCE_TOP_N,
            )
        except Exception as e:
            logger.error(f"Divergence analysis failed: {e}")
            results["errors"].append(f"Divergence analysis: {e}")

    # Sanity check: ensure latest daily data matches today
    data_ready = results["data_collected"]
    try:
        latest_date, record_count, active_count = _get_latest_price_snapshot()
        results["latest_data_date"] = str(latest_date) if latest_date else None
        results["latest_data_records"] = record_count
        results["active_stock_count"] = active_count
        if latest_date != datetime.now().date():
            data_ready = False
            msg = f"Latest daily_prices date is {latest_date}, expected {datetime.now().date()}"
            logger.error(msg)
            results["errors"].append(f"Data freshness: {msg}")
        if record_count == 0:
            data_ready = False
            msg = "No daily_prices records found for latest date"
            logger.error(msg)
            results["errors"].append(f"Data coverage: {msg}")
    except Exception as e:
        data_ready = False
        logger.error(f"Data readiness check failed: {e}")
        results["errors"].append(f"Data readiness: {e}")

    # Step 2: Update yesterday's prediction results
    logger.info("\n[Step 2] Updating prediction results...")
    try:
        predictor = RuleBasedPredictor()
        updated = predictor.update_actuals()
        results["actuals_updated"] = updated
        logger.info(f"Updated {updated} prediction results")
    except Exception as e:
        logger.error(f"Failed to update actuals: {e}")
        results["errors"].append(f"Update actuals: {e}")

    # Step 3: Run predictions
    logger.info("\n[Step 3] Running predictions...")
    if not data_ready:
        logger.error("Skipping predictions due to stale or missing daily data.")
        results["predictions_made"] = False
        results["errors"].append("Prediction skipped: data not ready")
        predictions = None
    else:
        try:
            predictor = RuleBasedPredictor()
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


def _is_within_orderbook_window(now: datetime) -> bool:
    """Return True when within configured orderbook collection window."""
    start = datetime.strptime(ORDERBOOK_START_TIME, "%H:%M").time()
    end = datetime.strptime(ORDERBOOK_END_TIME, "%H:%M").time()
    return start <= now.time() <= end


def run_orderbook_snapshot() -> dict:
    """Collect live orderbook snapshots (no historical backfill)."""
    results = {"status": "started", "errors": []}

    if not ORDERBOOK_COLLECTION_ENABLED:
        results["status"] = "disabled"
        return results

    if not is_trading_day():
        results["status"] = "skipped_non_trading_day"
        return results

    if not _is_within_orderbook_window(datetime.now()):
        results["status"] = "skipped_outside_window"
        return results

    try:
        init_db()
        orderbook_collector = OrderBookCollector()
        orderbook_stats = orderbook_collector.collect_and_save(show_progress=False)
        results["orderbook_stats"] = orderbook_stats
        results["status"] = "completed"
        logger.info(
            f"Orderbook: {orderbook_stats['success']} stocks, "
            f"{orderbook_stats['records']} records"
        )
    except Exception as e:
        logger.error(f"Orderbook collection failed: {e}")
        results["errors"].append(f"Orderbook: {e}")
        results["status"] = "failed"

    return results


def run_scheduled():
    """Run with scheduler - executes daily at specified time"""

    # Schedule for after market close (4:45 PM WIB / 09:45 UTC)
    schedule_time = "16:45"

    logger.info(f"Scheduler started. Will run daily at {schedule_time}")
    logger.info("Press Ctrl+C to stop.")

    schedule.every().day.at(schedule_time).do(run_daily_workflow)
    if ORDERBOOK_SCHEDULE_ENABLED:
        schedule.every(ORDERBOOK_INTERVAL_MINUTES).minutes.do(run_orderbook_snapshot)

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
    4. Warm-up complete (rule-based, no ML training)
    """
    from collectors import StockListCollector, HistoricalDataLoader

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

    # Step 4: No model training in rule-based system
    logger.info("\n[Step 4] Rule-based system ready (no ML training).")

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
