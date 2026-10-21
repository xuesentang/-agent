"""工具注册中心。

内置工具启动时登记,MCP 工具连上 Server 后动态发现、现问现拿,一律进同一份表。
新工具 register 进来即可被 bind_tools 用上,不改核心代码。

## 为什么元数据挂在表上而不是工具类上

「来源 / 是否写操作 / 来自哪个 Server」是**我们这个系统**的属性,不是 LangChain
工具协议的属性。塞进 `BaseTool` 子类,要么所有工具都得是我们的子类(MCP 来的
`StructuredTool` 得再包一层),要么改 `args_schema` 夹带私货。

所以:`BaseTool` 原样交给 `bind_tools`(协议不变),元数据另存一份 sidecar。
新增一类工具只多一条 meta,不动协议。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

SOURCE_BUILTIN = "builtin"
SOURCE_MCP = "mcp"


@dataclass(frozen=True)
class ToolMeta:
    """一条工具的登记信息。`read_only=False` 的工具要过确认闸。"""

    name: str
    source: str = SOURCE_BUILTIN
    read_only: bool = True
    server: str | None = None  # 来自哪个 MCP Server;内置工具为 None

    @property
    def is_write(self) -> bool:
        return not self.read_only


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metas: dict[str, ToolMeta] = {}

    def register(self, tool: BaseTool, meta: ToolMeta | None = None) -> None:
        """登记一个工具。重名直接报错——静默覆盖比报错更危险。"""
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}(重名说明配置错了,拒绝静默覆盖)")
        self._tools[tool.name] = tool
        self._metas[tool.name] = meta or ToolMeta(name=tool.name)

    def replace(self, tool: BaseTool, meta: ToolMeta | None = None) -> None:
        """同一个名字换实现(MCP 清单滚动刷新时用),不是「新登记」。"""
        if tool.name not in self._tools:
            self.register(tool, meta)
            return
        self._tools[tool.name] = tool
        self._metas[tool.name] = meta or self._metas[tool.name]

    def drop(self, name: str) -> None:
        self._tools.pop(name, None)
        self._metas.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def meta(self, name: str) -> ToolMeta | None:
        return self._metas.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def for_bind(self) -> list[BaseTool]:
        """给 llm.bind_tools 的清单(稳定顺序:按名排序,可复现)。"""
        return [self._tools[n] for n in self.names()]

    def specs(self) -> list[dict]:
        """给日志与前端看的清单:名字、用途、参数 Schema、来源、读写。"""
        out: list[dict] = []
        for name in self.names():
            tool = self._tools[name]
            meta = self._metas[name]
            schema_fn = getattr(getattr(tool, "args_schema", None), "model_json_schema", None)
            out.append(
                {
                    "name": name,
                    "description": (getattr(tool, "description", "") or "").strip(),
                    "parameters": schema_fn() if callable(schema_fn) else {},
                    "source": meta.source,
                    "server": meta.server,
                    "read_only": meta.read_only,
                }
            )
        return out

    def by_source(self, source: str) -> list[str]:
        return [n for n in self.names() if self._metas[n].source == source]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
