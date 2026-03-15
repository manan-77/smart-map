"""
UserKnowledge model — permanent per-user knowledge base.

Stores smart, LLM-discovered knowledge from conversations.
Entity types are dynamic — the LLM creates whatever type fits the pattern.
Safety filtering prevents false positives (e.g., inferring home from a single nav request).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, Index
from sqlalchemy.dialects.postgresql import UUID, JSON
from backend.database.db import Base


class UserKnowledge(Base):
    __tablename__ = "user_knowledge"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    knowledge_type = Column(String, nullable=False)       # dynamic — LLM creates freely
    key = Column(String, nullable=False)                   # unique identifier within type
    value = Column(JSON, nullable=False)                   # flexible structured data
    display_category = Column(String, default="general")   # personality|travel|places|preferences|patterns
    safety_level = Column(String, default="inferred")      # explicit|inferred
    confidence = Column(Float, default=0.5)                # 0.0-1.0, grows with repetition
    occurrence_count = Column(Integer, default=1)
    source_sessions = Column(JSON, default=list)           # session_ids that contributed
    last_used_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))  # last time injected into prompt
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_knowledge_user_type", "user_id", "knowledge_type"),
        Index("idx_knowledge_user_key", "user_id", "knowledge_type", "key", unique=True),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "knowledge_type": self.knowledge_type,
            "key": self.key,
            "value": self.value,
            "display_category": self.display_category,
            "safety_level": self.safety_level,
            "confidence": self.confidence,
            "occurrence_count": self.occurrence_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
