"""ToolRegistry 单测:登记、元数据(来源/读写/Server)、重名拒绝、滚动替换、清单可解释。"""

import pytest
from langchain_core.tools import tool

from app.core.tools.registry import SOURCE_BUILTIN, SOURCE_MCP, ToolMeta, ToolRegistry


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
    """重名说明配置错了(比如内置和 MCP 各报一个),静默覆盖更危险。"""
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    with pytest.raises(ValueError, match="重名"):
        reg.register(_make_tool("alpha"))


def test_meta_defaults_to_readonly_builtin():
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    meta = reg.meta("alpha")
    assert meta.source == SOURCE_BUILTIN
    assert meta.read_only is True
    assert meta.is_write is False


def test_write_tool_is_marked():
    reg = ToolRegistry()
    reg.register(_make_tool("w"), ToolMeta(name="w", read_only=False))
    assert reg.meta("w").is_write is True


def test_replace_swaps_implementation_for_same_name():
    """MCP 清单刷新时同名工具换实现,不是「新登记」。"""
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    reg.replace(_make_tool("alpha"), ToolMeta(name="alpha", source=SOURCE_MCP, server="logistics"))
    assert reg.meta("alpha").server == "logistics"
    assert reg.names() == ["alpha"]


def test_drop_removes_tool_and_meta():
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    reg.drop("alpha")
    assert "alpha" not in reg
    assert reg.meta("alpha") is None


def test_specs_carry_name_description_schema_and_source():
    """需求 1 那三样(名字、用途、JSON Schema)加来源与读写,一行一条给得出来。"""
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    reg.register(_make_tool("beta"), ToolMeta(name="beta", source=SOURCE_MCP, server="aftersale", read_only=False))
    specs = {s["name"]: s for s in reg.specs()}
    assert specs["alpha"]["description"].startswith("测试工具")
    assert "query" in specs["alpha"]["parameters"]["properties"]
    assert specs["beta"]["source"] == SOURCE_MCP
    assert specs["beta"]["server"] == "aftersale"
    assert specs["beta"]["read_only"] is False


def test_by_source_splits_builtin_and_mcp():
    reg = ToolRegistry()
    reg.register(_make_tool("alpha"))
    reg.register(_make_tool("beta"), ToolMeta(name="beta", source=SOURCE_MCP, server="aftersale"))
    assert reg.by_source(SOURCE_BUILTIN) == ["alpha"]
    assert reg.by_source(SOURCE_MCP) == ["beta"]
