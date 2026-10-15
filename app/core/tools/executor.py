"""工具执行引擎(spec 功能需求 3:参数 Schema 校验、执行错误处理、超时重试、结果回灌)。

错误分诊三类(spec §4):
1. 参数不合法   → ok=False,校验说明回灌模型,让它追问用户或重新组织调用
2. 查询落空     → 工具自身返回「未找到」文案,ok=True(业务空结果不算故障)
3. 真故障(异常) → ok=False,错误说明如实回灌

重试只给网络抖动这类暂时性故障(transient_errors 由调用方注入);
业务空结果不重试,写操作默认不自动重试(ch08 的规矩,本章先立骨架)。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.config import get_settings
from app.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """一次工具调用的结局;content 即将回灌给模型的文本。"""

    tool_call_id: str
    name: str
    ok: bool
    content: str


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float | None = None,
        transient_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._registry = registry
        self._timeout = timeout if timeout is not None else get_settings().tool_timeout_seconds
        self._transient_errors = transient_errors

    async def execute(self, tool_call: dict) -> ToolResult:
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id", "")
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(
                call_id, name, False,
                f"工具 {name!r} 不存在。可用工具: {', '.join(self._registry.names()) or '(无)'}",
            )

        # 1. 参数 Schema 校验(pydantic,args_schema 由 @tool 注解自动生成)
        try:
            tool.args_schema.model_validate(args)
        except ValidationError as exc:
            reasons = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            logger.info("tool %s validation rejected: %s", name, reasons)
            return ToolResult(call_id, name, False, f"参数不合法({reasons}),请向用户追问或修正后重试")

        # 2. 执行 + 超时(+ 暂时性故障重试一次)
        try:
            content = await self._invoke_with_retry(tool, tool_call)
        except asyncio.TimeoutError:
            logger.warning("tool %s timed out after %ss", name, self._timeout)
            return ToolResult(call_id, name, False, f"工具 {name} 执行超时(>{self._timeout}s)")
        except Exception as exc:  # 真故障:如实回给模型,不静默吞
            logger.exception("tool %s failed", name)
            return ToolResult(call_id, name, False, f"工具 {name} 执行失败:{exc}")

        return ToolResult(call_id, name, True, str(content))

    async def _invoke_with_retry(self, tool: BaseTool, tool_call: dict) -> str:
        attempts = 2 if self._transient_errors else 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                # StructuredTool.ainvoke(ToolCall dict) → ToolMessage(带 tool_call_id 对号)
                msg = await asyncio.wait_for(
                    tool.ainvoke(tool_call), timeout=self._timeout
                )
                return msg.content
            except self._transient_errors as exc:  # 网络抖动类:再试一次
                last_exc = exc
                logger.warning(
                    "tool %s transient error (attempt %d/%d): %s",
                    tool_call.get("name"), attempt + 1, attempts, exc,
                )
        raise last_exc  # type: ignore[misc]
