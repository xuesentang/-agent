"""ToolExecutor TDD(spec §4 错误分诊三类 / 超时 / 重试)。"""

import asyncio

import pytest

from langchain_core.tools import tool

from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry


def _registry_with(tools: list) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _call(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


@tool
def query_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


@tool
def flaky_tool(x: str) -> str:
    """总是失败。x: 任意"""
    raise RuntimeError("上游 500")


@tool
def slow_tool(x: str) -> str:
    """慢工具。x: 任意"""
    import time

    time.sleep(2)
    return "done"


async def test_happy_path():
    ex = ToolExecutor(_registry_with([query_logistics]))
    result = await ex.execute(_call("query_logistics", {"order_id": "1001"}))
    assert result.ok is True
    assert "北京转运中心" in result.content
    assert result.tool_call_id == "call_1"


async def test_unknown_tool():
    ex = ToolExecutor(_registry_with([query_logistics]))
    result = await ex.execute(_call("no_such_tool", {}))
    assert result.ok is False
    assert "不存在" in result.content


async def test_missing_required_arg_is_validation_reject():
    ex = ToolExecutor(_registry_with([query_logistics]))
    result = await ex.execute(_call("query_logistics", {}))  # 缺 order_id
    assert result.ok is False
    assert "参数不合法" in result.content


async def test_wrong_arg_type_is_validation_reject():
    ex = ToolExecutor(_registry_with([query_logistics]))
    result = await ex.execute(_call("query_logistics", {"order_id": 123}))
    # pydantic v2 严格模式:int → str 可能强转;这里用显式非标量值触发校验失败
    result2 = await ex.execute(_call("query_logistics", {"order_id": ["not", "a", "string"]}))
    assert result2.ok is False
    assert "参数不合法" in result2.content


async def test_tool_exception_is_fault_not_crash():
    ex = ToolExecutor(_registry_with([flaky_tool]))
    result = await ex.execute(_call("flaky_tool", {"x": "a"}))
    assert result.ok is False
    assert "执行失败" in result.content
    assert "500" in result.content


async def test_timeout_reported():
    ex = ToolExecutor(_registry_with([slow_tool]), timeout=0.1)
    result = await ex.execute(_call("slow_tool", {"x": "a"}))
    assert result.ok is False
    assert "超时" in result.content


async def test_transient_error_retried_once():
    class Transient(Exception):
        pass

    calls = {"n": 0}

    @tool
    def transient_tool(x: str) -> str:
        """时好时坏。x: 任意"""
        calls["n"] += 1
        if calls["n"] == 1:
            raise Transient("连接重置")
        return "第二次成功"

    ex = ToolExecutor(_registry_with([transient_tool]), transient_errors=(Transient,))
    result = await ex.execute(_call("transient_tool", {"x": "a"}))
    assert result.ok is True
    assert result.content == "第二次成功"
    assert calls["n"] == 2


async def test_no_retry_for_business_failure():
    # 非 transient 异常不重试:flaky_tool 只被调一次(review Minor 3: 补计数断言)。
    calls = {"n": 0}

    @tool
    def counting_fail(x: str) -> str:
        """计数失败。x: 任意"""
        calls["n"] += 1
        raise RuntimeError("业务失败")

    ex = ToolExecutor(_registry_with([counting_fail]))
    result = await ex.execute(_call("counting_fail", {"x": "a"}))
    assert result.ok is False
    assert calls["n"] == 1
