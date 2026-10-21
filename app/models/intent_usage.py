"""intent_usage 表：按意图维度的 token 花销（ch09 需求 2 的本地替代）。

需求原本说「意图打进 trace 元数据，去 Langfuse 里按意图看花销」。Langfuse 这章没装，
所以自己落一张表——回答的是同一个问题：哪类意图最烧钱。

**这张表的 DDL 需求没给**（它假设查 Langfuse 就够了），是本章补的。字段刻意做少：
只留回答「哪类意图烧钱」必需的，加字段要能说出它回答什么问题。

`token_source` 必须区分 `estimated` 与 `api`：当前只能估算（LangGraph custom 事件里
只有文本，ReAct 内部的 usage_metadata 没往外传），估算值和账单一起只看趋势可以，
当钱数看就是错的。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt

SOURCE_ESTIMATED = "estimated"
SOURCE_API = "api"
TOKEN_SOURCES = (SOURCE_ESTIMATED, SOURCE_API)


class IntentUsage(Base):
    __tablename__ = "intent_usage"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInt, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    intent: Mapped[str] = mapped_column(String(32), index=True)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    token_source: Mapped[str] = mapped_column(
        Enum(*TOKEN_SOURCES, name="token_source"), default=SOURCE_ESTIMATED
    )
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens or 0) + int(self.completion_tokens or 0)
