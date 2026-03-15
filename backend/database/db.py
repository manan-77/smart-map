"""
Database module — SQLAlchemy engine and session factory.

Agent state persistence is handled by the LangGraph PostgreSQL checkpointer
(see backend/persistence/checkpointer.py). This module manages the
conversations and user_knowledge tables via SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from backend.config import DATABASE_URL


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables (idempotent). Call on app startup."""
    # Import models so they register with Base.metadata
    import backend.models.conversation  # noqa: F401
    import backend.models.knowledge     # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Migrate: add new columns if they don't exist (dev convenience)
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE user_knowledge ADD COLUMN IF NOT EXISTS "
                "safety_level VARCHAR DEFAULT 'inferred'"
            ))
            conn.execute(text(
                "ALTER TABLE user_knowledge ADD COLUMN IF NOT EXISTS "
                "display_category VARCHAR DEFAULT 'general'"
            ))
            conn.execute(text(
                "ALTER TABLE user_knowledge ADD COLUMN IF NOT EXISTS "
                "last_used_at TIMESTAMPTZ DEFAULT NOW()"
            ))
            conn.commit()
        except Exception:
            pass  # Column already exists or table doesn't exist yet

    print("🗄️  Conversation & Knowledge tables created/verified")
