"""Database configuration and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import get_settings

settings = get_settings()


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url

# Create SQLAlchemy engine with increased pool size for parallel processing
engine = create_engine(
    _normalize_database_url(settings.database_url),
    pool_pre_ping=True,
    pool_size=20,  # Increased from 5 to 20 for parallel RAGAS/semantic tests
    max_overflow=30,  # Increased from 10 to 30 (total: 50 connections)
    pool_recycle=1800,  # Recycle connections every 30 min (remote connections can go stale faster)
    pool_timeout=30,  # Wait up to 30 seconds for a connection
    connect_args={
        "options": "-c statement_timeout=60000",  # 60s query timeout
        "keepalives": 1,
        "keepalives_idle": 30,       # Send keepalive after 30s idle
        "keepalives_interval": 10,   # Retry keepalive every 10s
        "keepalives_count": 5,       # Drop after 5 failed keepalives
    },
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    # Import all models to register them with Base
    from app.models import db_models, giskard_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
