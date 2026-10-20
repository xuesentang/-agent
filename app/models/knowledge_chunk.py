"""knowledge_chunks 表:知识库 chunk 原文权威源 (ch03 DDL)。

MySQL 是原文权威源;向量落 Milvus 集合 knowledge(chunk 主键与 Milvus 主键对齐),
双写状态机靠 vectorize_status:pending(待向量化)→ done(已向量化,回填 vector_id)。
category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    # 三格拼成向量化文本(spec D3):分类/上级标题路径 + 问法或本节标题 + 正文答案
    category: Mapped[str] = mapped_column(String(255), index=True)
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    # 以下四类元数据只存不进向量
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # DDL 是 TINYINT(1) NOT NULL DEFAULT 0;SQLAlchemy Boolean 在 MySQL 编译为 TINYINT(1)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(BigInt, nullable=True)
    next_chunk_id: Mapped[int | None] = mapped_column(BigInt, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"),
        default="pending",
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def embedding_text(self) -> str:
        """三格拼成一段做向量化(spec D3);元数据不进向量。"""
        parts = [self.category.strip(), self.questions.strip(), self.answer.strip()]
        return "\n".join(p for p in parts if p)
