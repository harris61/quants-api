"""
Configuration settings for Quants-API
Indonesian Stock Market ML Prediction System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models" / "saved"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATABASE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==================== API KEYS ====================
DATASAHAM_API_KEY = os.getenv("DATASAHAM_API_KEY", "sbk_a05e1010d49474dba301908a72499aa96c83ee8ef869c4b3")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ==================== DATABASE ====================
DATABASE_PATH = DATABASE_DIR / "quants.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ==================== MODEL SETTINGS ====================
# Label threshold for top gainer (10% = 0.10)
TOP_GAINER_THRESHOLD = 0.10

# Number of stocks to predict as top picks
TOP_PICKS_COUNT = 10

# Minimum data points required for training
MIN_TRAINING_SAMPLES = 100

# Train/test split ratio
TEST_SIZE = 0.2

# ==================== FEATURE SETTINGS ====================
# Lookback periods for features
RETURN_PERIODS = [1, 3, 5, 10, 20]  # days
MA_PERIODS = [5, 10, 20, 50]  # Moving averages
VOLATILITY_PERIOD = 20  # days

# ==================== DATA COLLECTION ====================
# Historical data to collect (in days)
HISTORICAL_DAYS = 365  # 1 year

# Rate limiting for API calls (seconds between calls)
API_RATE_LIMIT = 0.5

# ==================== SYMBOL FILTER ====================
# Only include normal equity tickers (IDX equities are 4 uppercase letters)
EQUITY_SYMBOL_REGEX = r"^[A-Z]{4}$"

# ==================== BACKTESTING ====================
# Walk-forward validation settings
BACKTEST_TRAIN_DAYS = 180  # 6 months training window
BACKTEST_TEST_DAYS = 30    # 1 month test window

# ==================== NOTIFICATIONS ====================
# Send notification only if prediction confidence above threshold
NOTIFICATION_CONFIDENCE_THRESHOLD = 0.5

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "quants.log"

# ==================== NEW DATA SOURCES ====================
# Broker summary collection
BROKER_COLLECTION_ENABLED = True

# Insider trading collection (less frequent)
INSIDER_COLLECTION_ENABLED = True
INSIDER_COLLECTION_DAY = 0  # Monday (0=Mon, 6=Sun)

# Intraday data settings
INTRADAY_COLLECTION_ENABLED = True
INTRADAY_INTERVAL = "1h"  # Hourly candles
INTRADAY_DAYS = 5  # Days of intraday data to keep

# Feature pipeline settings
INCLUDE_BROKER_FEATURES = True
INCLUDE_INSIDER_FEATURES = True
INCLUDE_INTRADAY_FEATURES = True
INCLUDE_MOVER_FEATURES = True

# Movers collection
MOVERS_COLLECTION_ENABLED = True

# Movers-based trading filter (regime filter)
MOVERS_FILTER_ENABLED = False
MOVERS_FILTER_TYPES = ["top_value", "top_volume", "top_frequency"]
