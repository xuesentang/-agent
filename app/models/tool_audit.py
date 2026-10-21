"""tool_audit_logs 表:每次工具调用留一条(第八章需求 5)。

内置与 MCP 工具都记;被权限拒、被校验拦的调用**同样落一条**——审计要能回答
「模型试过什么但没被允许」,只记成功的等于只记了一半。

## 为什么这张表不挂外键

审计写入不能被引用约束拦住:会话删了、工单没了,调用记录仍然要留着(它就是为
追责与排查存在的)。`conversation_id` 只建普通索引,值可能指向一个已经不存在的
会话,这是设计出来的。

## 为什么写审计失败不反拦工具执行

审计是旁路。表锁了、库慢了,不该让用户查不了物流。仓库层把异常吞成一条 ERROR
日志,调用方拿不到异常——这是唯一一处「故意吞异常」,写在这行注释里以免后人不解。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, BigInt

# 状态取值与建表语句的 ENUM 一致(本地 SQLite 存字符串,MySQL 存 ENUM)
STATUS_OK = "成功"
STATUS_FAILED = "失败"
STATUS_TIMEOUT = "超时"
STATUS_VALIDATION = "校验拦下"
STATUS_DENIED = "权限拒绝"


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BigInt, primary_key=True, autoincrement=True)
    # 无会话上下文的调用(脚本、定时任务)为 NULL
    conversation_id: Mapped[int | None] = mapped_column(BigInt, nullable=True, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    tool_source: Mapped[str] = mapped_column(String(16), default="builtin")  # builtin / mcp
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON 文本
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
