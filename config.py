"""
Configuration settings for Quants-API
Indonesian Stock Market Rule-Based Ranking System

Strategy: MA50 + Momentum + Foreign Flow
- Movers filter: Only trade stocks in top value/volume/frequency lists
- Hard filters: Above MA50, not overextended (15%), slope not falling
- Scoring: Momentum (32%) + Slope (23%) + Dist50 (18%) + Volume (17%) + Foreign (10%)
- Performance: 27.50% precision (2.7x better than random)

See docs/STRATEGY.md for full documentation.
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

# ==================== STRATEGY SETTINGS ====================
# Label threshold for top gainer (5% = 0.05)
TOP_GAINER_THRESHOLD = 0.05

# Number of stocks to predict as top picks
TOP_PICKS_COUNT = 3

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

# ==================== HOLIDAYS ====================
# Populate with IDX holidays in YYYY-MM-DD format.
IDX_HOLIDAYS = [
    # "2025-12-25",
    # "2026-01-01",
]

# ==================== NOTIFICATIONS ====================

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

# Movers collection
MOVERS_COLLECTION_ENABLED = True

# Movers-based trading filter (regime filter)
MOVERS_FILTER_ENABLED = True
MOVERS_FILTER_TYPES = ["top_value", "top_volume", "top_frequency"]

# ==================== RULE-BASED SETTINGS ====================
# MA50-only rule-based daily ranking
RULE_BASED_MODEL_NAME = "rule_ma50_v2"
RULE_MA_SLOW = 50
RULE_SLOPE_LOOKBACK = 5  # daily bars for slope calculation

# Filter thresholds (MA50-based)
RULE_DIST50_MIN = 0.0             # must be above MA50 (close > MA50)
RULE_DIST50_MAX = 0.15            # not too extended above MA50 (max 15%)
RULE_SLOPE_FLAT_MIN = -0.002      # MA50 slope must not be falling sharply

# Momentum settings (5-day return)
RULE_MOMENTUM_PERIOD = 5          # 5-day return
RULE_MOMENTUM_FLOOR = -0.05       # -5% momentum scores 0
RULE_MOMENTUM_CEIL = 0.10         # +10% momentum scores 1

# Volume filter settings
RULE_VOLUME_MA_PERIOD = 20
RULE_VOLUME_RATIO_FLOOR = 0.5     # volume_ratio below this scores 0
RULE_VOLUME_RATIO_CEIL = 1.5      # volume_ratio above this scores 1

# Slope score mapping
RULE_SLOPE_SCORE_FLOOR = -0.001
RULE_SLOPE_SCORE_CEIL = 0.005     # increased ceiling for better spread

# Foreign flow settings
RULE_FOREIGN_FLOOR = -1_000_000_000    # -1B IDR foreign_net scores 0
RULE_FOREIGN_CEIL = 10_000_000_000     # +10B IDR foreign_net scores 1

# Ranking score weights (sum = 100)
RULE_SCORE_WEIGHT_MOMENTUM = 32   # recent price action
RULE_SCORE_WEIGHT_SLOPE = 23      # MA50 trend direction
RULE_SCORE_WEIGHT_DIST50 = 18     # position above MA50
RULE_SCORE_WEIGHT_VOLUME = 17     # volume confirmation
RULE_SCORE_WEIGHT_FOREIGN = 10    # foreign flow signal (lower due to sparse data)
RULE_SCORE_DIST50_CAP = 0.10      # dist50 scoring cap
