from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class FAQ(Base):
    __tablename__ = "faq"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(255), unique=True)
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), default="guest")
    status: Mapped[str] = mapped_column(String(32), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'tool')", name="ck_messages_role"),
        Index("ix_messages_conversation_id_id", "conversation_id", "id"),
    )
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Ticket(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("source_key", "content_hash", name="uq_knowledge_source_content"),
        CheckConstraint("vectorize_status IN ('pending','done','retired','retired_clean')", name="ck_knowledge_vectorize_status"),
        Index("ix_knowledge_status", "vectorize_status"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class QAExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"
    __table_args__ = (
        Index("ix_qa_stage_batch", "batch_no"),
        Index("ix_qa_stage_status", "status"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="extracted")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
