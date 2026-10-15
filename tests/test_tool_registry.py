"""ToolRegistry TDD(spec 功能需求 3:注册管理)。"""

import pytest

from langchain_core.tools import tool

from app.core.tools.registry import ToolRegistry


def _make_tool(name: str):
    @tool(name)
    def dummy(query: str) -> str:
        """测试工具。query: 任意"""
        return f"{name}:{query}"

    return dummy


def test_register_get_for_bind():
    reg = ToolRegistry()
    t = _make_tool("alpha")
    reg.register(t)
    assert "alpha" in reg
    assert reg.get("alpha") is t
    assert reg.get("missing") is None
    assert reg.for_bind() == [t]
    assert reg.names() == ["alpha"]
    assert len(reg) == 1


def test_duplicate_name_rejected():
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    with pytest.raises(ValueError, match="重名"):
        reg.register(_make_tool("alpha"))
