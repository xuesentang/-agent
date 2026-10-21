"""MCP 接入单测:配置解析、TTL 缓存、失败保留旧清单、合进注册表。"""

from __future__ import annotations

from langchain_core.tools import tool

from app.core.tools.mcp_source import McpToolSource, parse_servers, refresh_registry
from app.core.tools.registry import SOURCE_MCP, ToolRegistry


def _mcp_tool(name: str):
    @tool(name)
    def remote(query: str) -> str:
        """远端工具。query: 参数"""
        return f"{name}:{query}"

    return remote


class FakeClient:
    def __init__(self, tools=None, error: Exception | None = None):
        self.tools = tools or []
        self.error = error
        self.calls = 0

    async def get_tools(self):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.tools)


def test_parse_servers_reads_name_url_pairs():
    assert parse_servers("logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp") == {
        "logistics": "http://127.0.0.1:8121/mcp",
        "aftersale": "http://127.0.0.1:8122/mcp",
    }
    assert parse_servers("") == {}
    assert parse_servers("broken-without-equals") == {}


async def test_refresh_without_servers_returns_empty():
    source = McpToolSource(servers={}, client_factory=lambda _s: FakeClient())
    assert await source.refresh() == []


async def test_list_is_cached_within_ttl():
    client = FakeClient([_mcp_tool("a")])
    source = McpToolSource(servers={"s": "http://x/mcp"}, ttl_seconds=60, client_factory=lambda _s: client)
    await source.refresh()
    await source.refresh()
    assert client.calls == 1, "TTL 内不该重复拉清单"


async def test_force_refresh_bypasses_cache():
    """Server 加了工具要能立刻看到(验收 3 靠这条)。"""
    client = FakeClient([_mcp_tool("a")])
    source = McpToolSource(servers={"s": "http://x/mcp"}, ttl_seconds=60, client_factory=lambda _s: client)
    await source.refresh()
    client.tools.append(_mcp_tool("b"))
    tools = await source.refresh(force=True)
    assert {t.name for t in tools} == {"a", "b"}
    assert client.calls == 2


async def test_failure_keeps_previous_list():
    """MCP Server 挂了,上次的清单还得留下——客服不能因为一个 Server 挂就瘫。"""
    good = FakeClient([_mcp_tool("a")])
    source = McpToolSource(servers={"s": "http://x/mcp"}, ttl_seconds=0, client_factory=lambda _s: good)
    await source.refresh()

    source._client_factory = lambda _s: FakeClient(error=RuntimeError("connection refused"))
    tools = await source.refresh(force=True)
    assert [t.name for t in tools] == ["a"]
    assert source.last_error and "connection refused" in source.last_error


async def test_refresh_registry_registers_mcp_tools_with_source():
    registry = ToolRegistry()
    client = FakeClient([_mcp_tool("query_logistics_track")])
    source = McpToolSource(servers={"logistics": "http://x/mcp"}, client_factory=lambda _s: client)
    names = await refresh_registry(registry, source, force=True)
    assert names == ["query_logistics_track"]
    meta = registry.meta("query_logistics_track")
    assert meta.source == SOURCE_MCP
    assert meta.server == "logistics"
    assert meta.read_only is True, "MCP 工具默认只读:对方的用途声明不可信,写操作由我们这侧另配"


async def test_multi_server_attributes_each_tool_to_its_server():
    """两个 Server 各拉一次,归属不能靠猜:谁拉出来的就记谁。"""

    def factory(servers):
        name = next(iter(servers))
        return FakeClient([_mcp_tool(f"{name}_tool")])

    registry = ToolRegistry()
    source = McpToolSource(
        servers={"logistics": "http://a/mcp", "aftersale": "http://b/mcp"},
        client_factory=factory,
    )
    names = await refresh_registry(registry, source, force=True)
    assert set(names) == {"logistics_tool", "aftersale_tool"}
    assert registry.meta("logistics_tool").server == "logistics"
    assert registry.meta("aftersale_tool").server == "aftersale"


async def test_refresh_registry_is_idempotent():
    registry = ToolRegistry()
    client = FakeClient([_mcp_tool("a")])
    source = McpToolSource(servers={"logistics": "http://x/mcp"}, client_factory=lambda _s: client)
    await refresh_registry(registry, source, force=True)
    await refresh_registry(registry, source, force=True)
    assert len(registry) == 1
