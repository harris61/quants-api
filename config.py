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
# API keys must be set via environment variables or .env file
DATASAHAM_API_KEY = os.getenv("DATASAHAM_API_KEY")
if not DATASAHAM_API_KEY:
    import warnings
    warnings.warn(
        "DATASAHAM_API_KEY not set. Please set it in .env file or environment variables. "
        "Get your API key from https://datasaham.io",
        UserWarning
    )
    DATASAHAM_API_KEY = ""  # Will fail gracefully when API is called

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ==================== DATABASE ====================
DATABASE_PATH = DATABASE_DIR / "quants.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# ==================== MODEL SETTINGS ====================
# Label threshold for top gainer (7% = 0.07)
TOP_GAINER_THRESHOLD = 0.07

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

# IDX trading days per year (differs from US 252)
# Indonesia has more holidays (Eid, Nyepi, etc.)
IDX_TRADING_DAYS_PER_YEAR = 242

# ==================== DATA COLLECTION ====================
# Historical data to collect (in days)
HISTORICAL_DAYS = 365  # 1 year

# Daily collection window (in days)
DAILY_COLLECT_DAYS = 30

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

# ==================== TRAINING SETTINGS ====================
# Include delisted stocks in training to avoid survivorship bias
# Set to True for more robust backtesting
INCLUDE_DELISTED_IN_TRAINING = True

# Portfolio diversification settings
MAX_SECTOR_CONCENTRATION = 0.4  # Max 40% of picks from one sector
MAX_SUBSECTOR_CONCENTRATION = 0.3  # Max 30% of picks from one subsector

# ==================== RULE-BASED SETTINGS ====================
# MA rule-based daily ranking (MA20/MA50)
RULE_BASED_MODEL_NAME = "rule_ma20_ma50_v1"
RULE_MA_FAST = 20
RULE_MA_SLOW = 50
RULE_SLOPE_LOOKBACK = 5  # daily bars

# Distance thresholds
RULE_DIST20_ENTRY = 0.04
RULE_DIST20_OVEREXTENDED = 0.08
RULE_DIST50_FALLING_KNIFE = -0.03

# Slope thresholds (flat/naik)
RULE_SLOPE_FLAT_MIN = -0.002
RULE_SLOPE_SCORE_FLOOR = -0.001
RULE_SLOPE_SCORE_CEIL = 0.001

# Ranking score weights (sum = 100)
RULE_SCORE_WEIGHT_PROX = 40
RULE_SCORE_WEIGHT_SLOPE = 25
RULE_SCORE_WEIGHT_DIST50 = 25
RULE_SCORE_WEIGHT_RECLAIM = 10
RULE_SCORE_DIST50_CAP = 0.06
