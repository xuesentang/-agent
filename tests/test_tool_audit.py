"""审计单测:成功/拦截都落一条、字段齐、写失败不反拦执行。"""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolMeta, ToolRegistry
from app.models.tool_audit import (
    STATUS_DENIED,
    STATUS_OK,
    STATUS_VALIDATION,
)
from app.repositories import audit_repo


@tool
def echo(text: str) -> str:
    """原样返回。text: 任意文本"""
    return f"echo:{text}"


@tool
def needs_number(count: int) -> str:
    """问一个数字。count: 数量"""
    return f"got {count}"


@tool
def create_thing(name: str) -> str:
    """写操作。name: 名字"""
    return f"created {name}"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(echo)
    reg.register(needs_number)
    reg.register(create_thing, ToolMeta(name="create_thing", read_only=False))
    return reg


async def test_successful_call_is_audited(session, session_factory):
    executor = ToolExecutor(_registry(), session_factory=session_factory)
    result = await executor.execute({"name": "echo", "args": {"text": "hi"}, "id": "c1"})
    assert result.ok and result.content == "echo:hi"
    await session.flush()

    from sqlalchemy import select

    from app.models.tool_audit import ToolAuditLog

    logs = list((await session.execute(select(ToolAuditLog))).scalars().all())
    assert len(logs) == 1
    log = logs[0]
    assert log.tool_name == "echo"
    assert log.tool_source == "builtin"
    assert log.tool_call_id == "c1"
    assert log.status == STATUS_OK
    assert log.arguments and "hi" in log.arguments
    assert log.result_summary == "echo:hi"
    assert log.duration_ms is not None and log.duration_ms >= 0


async def test_validation_rejection_is_audited(session, session_factory):
    """被校验拦下的调用同样要落——审计要能回答「模型试过什么」。"""
    executor = ToolExecutor(_registry(), session_factory=session_factory)
    result = await executor.execute({"name": "needs_number", "args": {"count": "不是数字"}, "id": "c2"})
    assert result.ok is False
    await session.flush()

    from sqlalchemy import select

    from app.models.tool_audit import ToolAuditLog

    log = (await session.execute(select(ToolAuditLog))).scalars().one()
    assert log.status == STATUS_VALIDATION
    assert log.error_message and "参数不合法" in log.error_message


async def test_permission_denial_is_audited(session, session_factory):
    executor = ToolExecutor(_registry(), session_factory=session_factory)
    result = await executor.execute({"name": "create_thing", "args": {"name": "x"}, "id": "c3"}, confirmed=False)
    assert result.ok is False and "未确认" in result.content
    await session.flush()

    from sqlalchemy import select

    from app.models.tool_audit import ToolAuditLog

    log = (await session.execute(select(ToolAuditLog))).scalars().one()
    assert log.status == STATUS_DENIED
    assert log.tool_name == "create_thing"


async def test_audit_failure_never_blocks_execution(engine, session, session_factory, monkeypatch):
    """审计表炸了也必须把工具结果交回去:审计是旁路。"""

    class BoomMaker:
        def __call__(self):
            raise RuntimeError("审计库挂了")

    executor = ToolExecutor(_registry(), session_factory=BoomMaker())
    result = await executor.execute({"name": "echo", "args": {"text": "ok"}, "id": "c4"})
    assert result.ok is True and result.content == "echo:ok"


async def test_no_session_factory_means_no_audit():
    """不传 session_factory(单测、脚本)就不落审计,不该报错。"""
    executor = ToolExecutor(_registry())
    result = await executor.execute({"name": "echo", "args": {"text": "x"}, "id": "c5"})
    assert result.ok is True


async def test_conversation_id_comes_from_contextvar(session, session_factory):
    from app.core.tools.context import current_conversation_id

    token = current_conversation_id.set(4242)
    try:
        executor = ToolExecutor(_registry(), session_factory=session_factory)
        await executor.execute({"name": "echo", "args": {"text": "x"}, "id": "c6"})
        await session.flush()
        from sqlalchemy import select

        from app.models.tool_audit import ToolAuditLog

        log = (await session.execute(select(ToolAuditLog))).scalars().one()
        assert log.conversation_id == 4242
    finally:
        current_conversation_id.reset(token)


@pytest.mark.parametrize("status_fn", [audit_repo.list_by_status])
async def test_list_by_status_filters(session, session_factory, status_fn):
    executor = ToolExecutor(_registry(), session_factory=session_factory)
    await executor.execute({"name": "echo", "args": {"text": "x"}, "id": "c7"})
    await executor.execute({"name": "create_thing", "args": {"name": "y"}, "id": "c8"}, confirmed=False)
    await session.flush()
    denied = await status_fn(session, STATUS_DENIED)
    assert [r.tool_name for r in denied] == ["create_thing"]
