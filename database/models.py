"""
SQLAlchemy Database Models for Quants-API
Indonesian Stock Market ML Prediction System
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime,
    Boolean, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Stock(Base):
    """Master table for stock symbols"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(255))
    sector_id = Column(String(50))
    sector_name = Column(String(100))
    listing_date = Column(Date)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    daily_prices = relationship("DailyPrice", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"


class DailyPrice(Base):
    """Daily OHLCV and trading data"""
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)

    # OHLCV Data
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    # Additional trading data
    value = Column(Float)           # Trading value (volume * price)
    frequency = Column(Integer)     # Number of transactions

    # Foreign flow data
    foreign_buy = Column(Float)     # Foreign buy value
    foreign_sell = Column(Float)    # Foreign sell value
    foreign_net = Column(Float)     # Net foreign (buy - sell)

    # Change data
    change = Column(Float)          # Price change
    change_percent = Column(Float)  # Percentage change

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="daily_prices")

    # Indexes for faster queries
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uix_stock_date'),
        Index('ix_daily_prices_date', 'date'),
        Index('ix_daily_prices_stock_date', 'stock_id', 'date'),
    )

    def __repr__(self):
        return f"<DailyPrice(stock_id={self.stock_id}, date='{self.date}', close={self.close})>"


class Prediction(Base):
    """Model predictions and their outcomes"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    prediction_date = Column(Date, nullable=False)  # Date when prediction was made
    target_date = Column(Date, nullable=False)      # Date being predicted (T+1)

    # Prediction data
    probability = Column(Float, nullable=False)     # Predicted probability of top gainer
    rank = Column(Integer)                          # Rank among all predictions

    # Actual outcome (filled after target_date)
    actual_return = Column(Float)                   # Actual return on target_date
    is_top_gainer = Column(Boolean)                 # Was it actually a top gainer?
    is_correct = Column(Boolean)                    # Was prediction correct?

    # Model metadata
    model_version = Column(String(50))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")

    # Indexes
    __table_args__ = (
        UniqueConstraint('stock_id', 'prediction_date', name='uix_stock_prediction_date'),
        Index('ix_predictions_date', 'prediction_date'),
        Index('ix_predictions_target_date', 'target_date'),
    )

    def __repr__(self):
        return f"<Prediction(stock_id={self.stock_id}, date='{self.prediction_date}', prob={self.probability:.2f})>"


class MarketData(Base):
    """Daily market-wide data (IHSG, sector performance, etc.)"""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    date = Column(Date, unique=True, nullable=False, index=True)

    # IHSG data
    ihsg_open = Column(Float)
    ihsg_high = Column(Float)
    ihsg_low = Column(Float)
    ihsg_close = Column(Float)
    ihsg_change_percent = Column(Float)

    # Market breadth
    advancing_stocks = Column(Integer)   # Number of stocks going up
    declining_stocks = Column(Integer)   # Number of stocks going down
    unchanged_stocks = Column(Integer)   # Number of unchanged stocks

    # Market totals
    total_value = Column(Float)          # Total market trading value
    total_volume = Column(Float)         # Total market volume
    total_frequency = Column(Integer)    # Total transactions

    # Foreign flow totals
    total_foreign_buy = Column(Float)
    total_foreign_sell = Column(Float)
    total_foreign_net = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MarketData(date='{self.date}', ihsg_close={self.ihsg_close})>"


class ModelMetrics(Base):
    """Track model performance over time"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    model_version = Column(String(50), nullable=False)

    # Performance metrics
    precision_at_10 = Column(Float)      # Precision@10
    hit_rate = Column(Float)             # % days with at least 1 correct
    avg_return = Column(Float)           # Average return of predictions
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)

    # Rolling metrics (last 30 days)
    rolling_precision = Column(Float)
    rolling_hit_rate = Column(Float)
    rolling_avg_return = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('date', 'model_version', name='uix_date_model'),
        Index('ix_model_metrics_date', 'date'),
    )

    def __repr__(self):
        return f"<ModelMetrics(date='{self.date}', precision@10={self.precision_at_10:.2f})>"


class CorporateAction(Base):
    """Corporate actions (dividends, stock splits, etc.)"""
    __tablename__ = "corporate_actions"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    symbol = Column(String(10), nullable=False, index=True)

    action_type = Column(String(50), nullable=False)  # dividend, stocksplit, rights, rups

    # Dates
    announcement_date = Column(Date)
    ex_date = Column(Date)
    record_date = Column(Date)
    payment_date = Column(Date)

    # Details
    value = Column(Float)            # Dividend amount or split ratio
    value_formatted = Column(String(100))
    description = Column(String(500))

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('ix_corporate_actions_symbol', 'symbol'),
        Index('ix_corporate_actions_ex_date', 'ex_date'),
        Index('ix_corporate_actions_type', 'action_type'),
    )

    def __repr__(self):
        return f"<CorporateAction(symbol='{self.symbol}', type='{self.action_type}', ex_date='{self.ex_date}')>"
