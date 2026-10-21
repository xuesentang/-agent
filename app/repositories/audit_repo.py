"""工具调用审计(第八章需求 5)。

一条调用一条记录,包括被拦下的那些(校验不过、权限拒绝)。

## 唯一的「故意吞异常」

`record()` 把写库异常吃成一条 ERROR 日志,不往外抛。理由是审计是旁路:表锁了、
库慢了,不该让用户查不了物流。调用方(g 执行引擎)拿不到异常,也不需要处理。

这条规矩只在审计这一处成立,别照着写到别的地方去。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select

from app.core.tools.context import current_conversation_id
from app.models.tool_audit import ToolAuditLog

logger = logging.getLogger(__name__)

# 审计里存的结果摘要上限:表是拿来排查的,不是拿来存全文的
RESULT_SUMMARY_LIMIT = 500


def _dump_args(arguments: Any) -> str | None:
    if arguments in (None, {}, ""):
        return None
    try:
        return json.dumps(arguments, ensure_ascii=False)[:RESULT_SUMMARY_LIMIT]
    except (TypeError, ValueError):
        return str(arguments)[:RESULT_SUMMARY_LIMIT]


async def record(
    session_factory,
    *,
    tool_name: str,
    tool_source: str = "builtin",
    tool_call_id: str | None = None,
    mcp_server: str | None = None,
    arguments: Any = None,
    result: str | None = None,
    status: str = "成功",
    error_message: str | None = None,
    retry_count: int = 0,
    duration_ms: int | None = None,
    conversation_id: int | None = None,
) -> None:
    """落一条审计。任何异常只记日志,绝不影响工具执行。"""
    cid = conversation_id if conversation_id is not None else current_conversation_id.get()
    try:
        async with session_factory() as session:
            session.add(
                ToolAuditLog(
                    conversation_id=cid,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_source=tool_source,
                    mcp_server=mcp_server,
                    arguments=_dump_args(arguments),
                    result_summary=(result or "")[:RESULT_SUMMARY_LIMIT] or None,
                    status=status,
                    error_message=(error_message or "")[:512] or None,
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 - 审计是旁路,不许反拦执行
        logger.exception("审计写入失败(tool=%s status=%s),已忽略", tool_name, status)


async def list_by_conversation(session, conversation_id: int, limit: int = 50) -> list[ToolAuditLog]:
    stmt = (
        select(ToolAuditLog)
        .where(ToolAuditLog.conversation_id == conversation_id)
        .order_by(ToolAuditLog.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def list_by_status(session, status: str, limit: int = 50) -> list[ToolAuditLog]:
    stmt = (
        select(ToolAuditLog)
        .where(ToolAuditLog.status == status)
        .order_by(ToolAuditLog.id.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())
