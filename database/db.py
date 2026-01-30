"""
Database connection and session management for Quants-API
"""

from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_URL
from database.models import Base


# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set True for SQL debugging
    connect_args={
        "check_same_thread": False,  # Required for SQLite
        "timeout": 30,  # seconds to wait on locked database
    }
)

# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Initialize database - create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        _ensure_broker_category_column()
        _ensure_stock_market_cap_columns()
    except OperationalError as exc:
        if "already exists" not in str(exc):
            raise
    print(f"Database initialized at: {DATABASE_URL}")


def _ensure_broker_category_column() -> None:
    """Add broker_category column to broker_summaries if missing (SQLite)."""
    with engine.begin() as conn:
        table = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='broker_summaries'")
        ).fetchone()
        if not table:
            return
        cols = conn.execute(text("PRAGMA table_info('broker_summaries')")).fetchall()
        col_names = {row[1] for row in cols}
        if "broker_category" in col_names:
            return
        conn.execute(text("ALTER TABLE broker_summaries ADD COLUMN broker_category VARCHAR(50)"))


def _ensure_stock_market_cap_columns() -> None:
    """Add market cap columns to stocks if missing (SQLite)."""
    with engine.begin() as conn:
        table = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        ).fetchone()
        if not table:
            return
        cols = conn.execute(text("PRAGMA table_info('stocks')")).fetchall()
        col_names = {row[1] for row in cols}
        if "market_cap" not in col_names:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN market_cap FLOAT"))
        if "market_cap_formatted" not in col_names:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN market_cap_formatted VARCHAR(100)"))
        if "market_cap_currency" not in col_names:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN market_cap_currency VARCHAR(10)"))
        if "market_cap_updated_at" not in col_names:
            conn.execute(text("ALTER TABLE stocks ADD COLUMN market_cap_updated_at DATETIME"))


def drop_db():
    """Drop all tables - USE WITH CAUTION"""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped!")


def get_session() -> Session:
    """Get a new database session"""
    return SessionLocal()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            session.add(new_stock)
            # auto-commits on success, auto-rollbacks on exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Convenience functions for common operations

def get_stock_by_symbol(session: Session, symbol: str):
    """Get stock by symbol"""
    from database.models import Stock
    return session.query(Stock).filter(Stock.symbol == symbol.upper()).first()


def get_or_create_stock(session: Session, symbol: str, name: str = None, **kwargs):
    """Get existing stock or create new one"""
    from database.models import Stock

    stock = get_stock_by_symbol(session, symbol)
    if stock is None:
        stock = Stock(symbol=symbol.upper(), name=name, **kwargs)
        session.add(stock)
        session.flush()
    return stock


def bulk_upsert_daily_prices(session: Session, stock_id: int, price_data: list):
    """Bulk insert or update daily price records

    Args:
        session: Database session
        stock_id: Stock ID
        price_data: List of dicts with price data
    """
    from datetime import datetime
    from sqlalchemy.dialects.sqlite import insert
    from database.models import DailyPrice

    if not price_data:
        return

    # Prepare data with stock_id
    records = []
    for data in price_data:
        record = {
            "stock_id": stock_id,
            "date": data["date"] if isinstance(data["date"], datetime) else datetime.strptime(data["date"], "%Y-%m-%d").date(),
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "close": data.get("close"),
            "volume": data.get("volume"),
            "value": data.get("value"),
            "frequency": data.get("frequency"),
            "foreign_buy": data.get("foreign_buy"),
            "foreign_sell": data.get("foreign_sell"),
            "foreign_net": data.get("foreign_net"),
            "change": data.get("change"),
            "change_percent": data.get("change_percent"),
        }
        records.append(record)

    # SQLite upsert (INSERT OR REPLACE)
    for record in records:
        stmt = insert(DailyPrice).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["stock_id", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "value": stmt.excluded.value,
                "frequency": stmt.excluded.frequency,
                "foreign_buy": stmt.excluded.foreign_buy,
                "foreign_sell": stmt.excluded.foreign_sell,
                "foreign_net": stmt.excluded.foreign_net,
                "change": stmt.excluded.change,
                "change_percent": stmt.excluded.change_percent,
            }
        )
        session.execute(stmt)


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
