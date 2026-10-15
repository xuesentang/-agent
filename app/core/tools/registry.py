"""工具注册中心(spec 功能需求 3:注册管理)。

内置工具启动时登记;ch08 起 MCP 工具动态发现后现问现拿,走同一份注册表。
新工具 register 进来即可被 bind_tools 用上,不改核心代码。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}(ch08 起重名即冲突,拒绝静默覆盖)")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def for_bind(self) -> list[BaseTool]:
        """给 llm.bind_tools 的清单(稳定顺序:按名排序,可复现)。"""
        return [self._tools[n] for n in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
