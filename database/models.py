"""
SQLAlchemy Database Models for Quants-API
Indonesian Stock Market ML Prediction System
"""

from datetime import datetime as dt
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
    created_at = Column(DateTime, default=dt.utcnow)
    updated_at = Column(DateTime, default=dt.utcnow, onupdate=dt.utcnow)

    # Relationships
    daily_prices = relationship("DailyPrice", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    broker_summaries = relationship("BrokerSummary", back_populates="stock", cascade="all, delete-orphan")
    insider_trades = relationship("InsiderTrade", back_populates="stock", cascade="all, delete-orphan")
    intraday_prices = relationship("IntradayPrice", back_populates="stock", cascade="all, delete-orphan")

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
    created_at = Column(DateTime, default=dt.utcnow)

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
    actual_return = Column(Float)                   # Actual intraday return (open -> close) on target_date
    is_top_gainer = Column(Boolean)                 # Was it actually a top gainer?
    is_correct = Column(Boolean)                    # Was prediction correct?

    # Model metadata
    model_version = Column(String(50))

    # Timestamps
    created_at = Column(DateTime, default=dt.utcnow)
    updated_at = Column(DateTime, default=dt.utcnow, onupdate=dt.utcnow)

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

    created_at = Column(DateTime, default=dt.utcnow)

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

    created_at = Column(DateTime, default=dt.utcnow)

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

    created_at = Column(DateTime, default=dt.utcnow)

    __table_args__ = (
        Index('ix_corporate_actions_symbol', 'symbol'),
        Index('ix_corporate_actions_ex_date', 'ex_date'),
        Index('ix_corporate_actions_type', 'action_type'),
    )

    def __repr__(self):
        return f"<CorporateAction(symbol='{self.symbol}', type='{self.action_type}', ex_date='{self.ex_date}')>"


class BrokerSummary(Base):
    """Daily broker activity per stock"""
    __tablename__ = "broker_summaries"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Broker identification
    broker_code = Column(String(10), nullable=False)
    broker_name = Column(String(100))

    # Activity data
    buy_value = Column(Float)        # Total buy value
    sell_value = Column(Float)       # Total sell value
    net_value = Column(Float)        # Net (buy - sell)
    buy_volume = Column(Float)       # Total buy volume
    sell_volume = Column(Float)      # Total sell volume
    net_volume = Column(Float)       # Net volume
    buy_frequency = Column(Integer)  # Number of buy transactions
    sell_frequency = Column(Integer) # Number of sell transactions

    created_at = Column(DateTime, default=dt.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="broker_summaries")

    __table_args__ = (
        UniqueConstraint('stock_id', 'date', 'broker_code', name='uix_stock_date_broker'),
        Index('ix_broker_summaries_stock_date', 'stock_id', 'date'),
        Index('ix_broker_summaries_broker', 'broker_code'),
        Index('ix_broker_summaries_date', 'date'),
    )

    def __repr__(self):
        return f"<BrokerSummary(stock_id={self.stock_id}, date='{self.date}', broker='{self.broker_code}')>"


class InsiderTrade(Base):
    """Insider trading transactions"""
    __tablename__ = "insider_trades"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)

    # Insider identification
    insider_name = Column(String(255), nullable=False)
    position = Column(String(100))       # Position/role in company
    relationship_type = Column(String(100))   # Relationship to company

    # Transaction details
    transaction_type = Column(String(20), nullable=False)  # 'buy' or 'sell'
    transaction_date = Column(Date, nullable=False)
    shares = Column(Float)               # Number of shares
    price = Column(Float)                # Transaction price
    value = Column(Float)                # Total value (shares * price)

    # Ownership after transaction
    shares_after = Column(Float)         # Shares owned after transaction

    # Metadata
    announcement_date = Column(Date)     # Date announcement was made
    created_at = Column(DateTime, default=dt.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="insider_trades")

    __table_args__ = (
        Index('ix_insider_trades_stock', 'stock_id'),
        Index('ix_insider_trades_date', 'transaction_date'),
        Index('ix_insider_trades_type', 'transaction_type'),
        Index('ix_insider_trades_insider', 'insider_name'),
    )

    def __repr__(self):
        return f"<InsiderTrade(stock_id={self.stock_id}, insider='{self.insider_name}', type='{self.transaction_type}')>"


class IntradayPrice(Base):
    """Hourly OHLCV data"""
    __tablename__ = "intraday_prices"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    datetime = Column(DateTime, nullable=False)  # Full timestamp
    date = Column(Date, nullable=False)          # Date part for easier querying
    hour = Column(Integer, nullable=False)       # Hour (9-16 for IDX)

    # OHLCV data
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    value = Column(Float)

    created_at = Column(DateTime, default=dt.utcnow)

    # Relationships
    stock = relationship("Stock", back_populates="intraday_prices")

    __table_args__ = (
        UniqueConstraint('stock_id', 'datetime', name='uix_stock_datetime'),
        Index('ix_intraday_prices_stock_date', 'stock_id', 'date'),
        Index('ix_intraday_prices_datetime', 'datetime'),
        Index('ix_intraday_prices_date', 'date'),
    )

    def __repr__(self):
        return f"<IntradayPrice(stock_id={self.stock_id}, datetime='{self.datetime}', close={self.close})>"


class DailyMover(Base):
    """Daily mover lists (top value/volume/frequency/gainer/loser/foreign flow)"""
    __tablename__ = "daily_movers"

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Type of mover list (e.g. top_value, top_volume, top_frequency, top_gainer, top_loser, net_foreign_buy, net_foreign_sell)
    mover_type = Column(String(50), nullable=False)

    # Rank within the mover list
    rank = Column(Integer)

    # Optional score (percent, value, volume, etc.)
    score = Column(Float)

    created_at = Column(DateTime, default=dt.utcnow)

    # Relationships
    stock = relationship("Stock")

    __table_args__ = (
        UniqueConstraint('stock_id', 'date', 'mover_type', name='uix_mover_stock_date_type'),
        Index('ix_daily_movers_stock_date', 'stock_id', 'date'),
        Index('ix_daily_movers_date', 'date'),
        Index('ix_daily_movers_type', 'mover_type'),
    )

    def __repr__(self):
        return f"<DailyMover(stock_id={self.stock_id}, date='{self.date}', type='{self.mover_type}')>"
