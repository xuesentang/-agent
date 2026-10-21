"""MCP 工具接入:连上 MCP Server 拉清单,拿到就能用。

## 为什么是「拉一次 + TTL 缓存」而不是常驻 session

`MultiServerMCPClient.get_tools()` 的语义是:为每个 Server 建一个临时 session
拉清单,工具调用时再建新 session。也就是说清单本身就是一次性快照。

这正好对上需求里那句「Server 加个工具、只重启 Server,客服系统不重启也能用上」:
按 TTL 重新拉一次清单即可。常驻 session 反而更脆——session 绑着事件循环,
FastAPI 里换循环、热重载都会把它废掉。

## 失败保留上次清单

MCP Server 挂了不该让整个客服瘫痪:上一份清单里的工具还在,调用会报连接错,
但内置工具(查单、知识库)照常。所以拉取失败 = 打日志 + 返回旧清单,不抛。
"""

from __future__ import annotations

import logging
import time

from app.config import get_settings
from app.core.tools.registry import SOURCE_MCP, ToolMeta, ToolRegistry

logger = logging.getLogger(__name__)


def _is_loopback(url: str) -> bool:
    return "://127.0.0.1" in url or "://localhost" in url or "://[::1]" in url


def _loopback_client(headers=None, timeout=None, auth=None, **kwargs):
    """给本机 MCP Server 用的 httpx 客户端:`trust_env=False`,无视代理设置。"""
    import httpx

    return httpx.AsyncClient(
        headers=headers, timeout=timeout, auth=auth, trust_env=False, **kwargs
    )


def parse_servers(raw: str) -> dict[str, str]:
    """解析 `名字=URL,名字=URL`。空串返回空字典(等于不接)。"""
    out: dict[str, str] = {}
    for chunk in (raw or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            logger.warning("MCP_SERVERS 配置项 %r 缺少 '=',已忽略", item)
            continue
        name, url = item.split("=", 1)
        name, url = name.strip(), url.strip()
        if name and url:
            out[name] = url
    return out


class McpToolSource:
    """按 TTL 刷新的 MCP 工具源。`client_factory` 可注入(单测用假 client)。"""

    def __init__(
        self,
        servers: dict[str, str] | None = None,
        *,
        ttl_seconds: int | None = None,
        client_factory=None,
    ) -> None:
        settings = get_settings()
        self._servers = servers if servers is not None else parse_servers(settings.mcp_servers)
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.mcp_tool_ttl_seconds
        self._client_factory = client_factory
        self._tools: list = []
        self._tool_server: dict[str, str | None] = {}
        self._fetched_at: float = 0.0
        self._last_error: str | None = None

    @property
    def servers(self) -> dict[str, str]:
        return dict(self._servers)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _client(self, servers: dict[str, str] | None = None):
        """按给定 Server 集合造客户端。多 Server 时一 Server 一个客户端,见 `_fetch`。"""
        servers = self._servers if servers is None else servers
        if self._client_factory is not None:
            return self._client_factory(servers)
        from langchain_mcp_adapters.client import MultiServerMCPClient

        connections: dict[str, dict] = {}
        for name, url in servers.items():
            conn: dict = {"transport": "http", "url": url}
            if _is_loopback(url):
                # 本机回环绝不能走系统代理:机器上开着 HTTP_PROXY 时,发往 127.0.0.1 的
                # 请求会被代理接走,表现是「MCP 502 Bad Gateway」,而 Server 明明是好的。
                conn["httpx_client_factory"] = _loopback_client
            connections[name] = conn
        return MultiServerMCPClient(connections)

    async def _fetch(self) -> list:
        """拉清单,顺带记住每个工具来自哪个 Server。

        多 Server 时**分 Server 拉**:适配器不告诉我们工具归属,靠工具名的前缀去猜
        等于猜配置;分 Server 各拉一次,归属天然确定。Server 一般只有两三个、TTL 30 秒
        一次,多几个请求换一个确定的来源字段,值。
        """
        if len(self._servers) <= 1:
            tools = await self._client().get_tools()
            only = next(iter(self._servers), None)
            for t in tools:
                self._tool_server[t.name] = only
            return list(tools)

        merged: list = []
        for name, url in self._servers.items():
            part = await self._client({name: url}).get_tools()
            for t in part:
                if t.name in self._tool_server and self._tool_server[t.name] != name:
                    logger.warning(
                        "MCP 工具重名:%s 同时出现在 %s 与 %s,后者覆盖前者",
                        t.name, self._tool_server[t.name], name,
                    )
                self._tool_server[t.name] = name
            merged.extend(part)
        return merged

    def _expired(self) -> bool:
        return (time.monotonic() - self._fetched_at) >= self._ttl

    async def refresh(self, *, force: bool = False) -> list:
        """拉清单。未过期且非强制 → 直接返回缓存;拉取失败 → 返回上次的清单。"""
        if not self._servers:
            return []
        if not force and self._tools and not self._expired():
            return self._tools
        try:
            tools = await self._fetch()
        except Exception as exc:  # noqa: BLE001 - MCP 挂了不能拖垮客服
            self._last_error = str(exc)
            logger.warning(
                "MCP 工具清单拉取失败(保留上次 %s 个):%s", len(self._tools), exc
            )
            return self._tools
        self._tools = list(tools)
        self._fetched_at = time.monotonic()
        self._last_error = None
        logger.info(
            "MCP 工具清单已刷新:%s 个(%s)",
            len(self._tools),
            ", ".join(sorted(t.name for t in self._tools)) or "无",
        )
        return self._tools

    def server_of(self, tool) -> str | None:
        """工具来自哪个 Server:优先用分 Server 拉取时记下的归属。"""
        return self._tool_server.get(getattr(tool, "name", ""))


def _server_of(client_servers: dict[str, str], tool, source: McpToolSource | None = None) -> str | None:
    """判定工具来自哪个 Server。

    首选分 Server 拉取时记下的归属(见 `McpToolSource._fetch`)。退路:单 Server 时
    直接归属它;多 Server 且没记下时,看工具自带元数据有没有 Server 名,取不到就 None。
    """
    if source is not None:
        got = source.server_of(tool)
        if got:
            return got
    if len(client_servers) == 1:
        return next(iter(client_servers))
    meta = getattr(tool, "metadata", None) or {}
    if isinstance(meta, dict):
        for key in ("server", "mcp_server", "server_name"):
            if meta.get(key):
                return str(meta[key])
    return None


async def refresh_registry(
    registry: ToolRegistry, source: McpToolSource, *, force: bool = False
) -> list[str]:
    """把 MCP 清单合进注册表,返回本次登记/替换的工具名。

    合并不做「先删后加」:Server 侧删掉的工具留在表里会调不通,但静默删除正在被
    模型引用的工具更糟。要下线一个工具,改配置或改 Server 后重启客服服务。
    """
    tools = await source.refresh(force=force)
    names: list[str] = []
    for tool in tools:
        server = _server_of(source.servers, tool, source)
        registry.replace(tool, ToolMeta(name=tool.name, source=SOURCE_MCP, server=server))
        names.append(tool.name)
    return names


_source: McpToolSource | None = None


def get_mcp_source() -> McpToolSource:
    """进程级单例:TTL 缓存挂在它身上,每轮新建就失去了缓存的意义。"""
    global _source
    if _source is None:
        _source = McpToolSource()
    return _source
