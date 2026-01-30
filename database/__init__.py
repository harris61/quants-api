"""Database package for Quants-API"""

from database.db import (
    init_db,
    get_session,
    session_scope,
    get_stock_by_symbol,
    get_or_create_stock,
    bulk_upsert_daily_prices,
)
from database.models import (
    Base,
    Stock,
    DailyPrice,
    Prediction,
    MarketData,
    ModelMetrics,
    CorporateAction,
    BrokerSummary,
    InsiderTrade,
    MarketCapHistory,
    IntradayPrice,
)

__all__ = [
    # Database functions
    "init_db",
    "get_session",
    "session_scope",
    "get_stock_by_symbol",
    "get_or_create_stock",
    "bulk_upsert_daily_prices",
    # Models
    "Base",
    "Stock",
    "DailyPrice",
    "Prediction",
    "MarketData",
    "ModelMetrics",
    "CorporateAction",
    "BrokerSummary",
    "InsiderTrade",
    "MarketCapHistory",
    "IntradayPrice",
]
