"""Langfuse 可观测性(CH09):一轮对话一条 trace,意图/路由进元数据。

设计前提:**观测是旁路,不是功能**。LANGFUSE_ENABLED=false / key 缺失 /
SDK 导入失败 / host 不通,四种情况全部返回空 callbacks,聊天链路零感知、
零报错、零延迟差异。任何初始化失败只记日志,不抛异常。

一轮对话 = 一条 trace:
- 根 span 名字 `chat:{conversation_id}`,会话按 `langfuse_session_id` 分组;
- LangChain 世界(图节点 / LLM generation / 工具调用)由 SDK 的
  `langchain.CallbackHandler` 自动铺成 span 树;
- 意图/路由是图跑出来的,`intent` 事件到达时补写 trace 元数据;
- 拒答轮(`finish_reason=abstain`)补 `abstain` tag,UI 里按 tag 过滤。

v5 SDK 没有 `update_current_trace`:trace 级属性就是根 span 上的
`langfuse.trace.metadata.*` / `langfuse.trace.tags`,由
`propagate_attributes` 负责写。所以补写入口用 `use_span` 把根 span 设为
当前 span,保证落点在根上,不依赖此刻恰好哪个 span 活跃。
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any, Iterator, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# Langfuse 对 session_id / trace_name 的约束:US-ASCII 且 ≤200 字符。
# session_id 是前端传的任意串,不合规时 SDK 只会告警丢弃——这里先过滤,
# 免得每轮刷一条 warning。
_MAX_ATTR_LEN = 200


def _valid_attr(value: Any) -> Optional[str]:
    """合规的 trace 属性值;不合规返回 None(SDK 对 None 直接跳过)。"""
    if isinstance(value, str) and value.isascii() and 0 < len(value) <= _MAX_ATTR_LEN:
        return value
    return None


def build_trace_metadata(*, conversation_id: int, session_id: str) -> dict[str, str]:
    """LangChain config 的 metadata。SDK 的 handler 认 `langfuse_*` 这几个键:
    session_id 把同一会话的 trace 归组,trace_name 给 trace 起名。"""
    meta = {"langfuse_trace_name": f"chat:{conversation_id}"}
    session = _valid_attr(session_id)
    if session:
        meta["langfuse_session_id"] = session
    return meta


def _probe_host(host: str, timeout: float = 3.0) -> bool:
    """host 健康检查(GET /api/public/health)。不通就当没配,聊天照常。"""
    import httpx

    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/public/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _load_sdk():
    """导入 SDK。返回 (Langfuse, CallbackHandler);没装/不完整返回 None。"""
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        return Langfuse, CallbackHandler
    except Exception:
        logger.info("langfuse SDK 不可用,trace 上报关闭", exc_info=True)
        return None


def _build_client(settings):
    """按 settings 显式构造 client——不改进程环境变量,配置只认 .env 一份。"""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
        tracing_enabled=True,
    )


@lru_cache(maxsize=4)
def _callbacks_for(enabled: bool, public_key: str, secret_key: str, host: str) -> list:
    """四种降级路径全部返回 [];成功返回 [CallbackHandler]。

    缓存键带上配置本身:测试里改 settings 不会读到上一件的缓存。
    """
    if not enabled:
        return []
    if not (public_key and secret_key):
        logger.info("Langfuse key 未配置,trace 上报关闭")
        return []
    if not _probe_host(host):
        logger.warning("Langfuse host 不可达(%s),trace 上报关闭", host)
        return []
    sdk = _load_sdk()
    if sdk is None:
        return []
    try:
        settings = get_settings()
        # 先建 client 再建 handler:handler 按 public_key 找同一个实例
        _build_client(settings)
        return [sdk[1](public_key=public_key)]
    except Exception:
        logger.exception("Langfuse 初始化失败,trace 上报关闭")
        return []


def get_langfuse_callbacks() -> list:
    """LangChain callbacks。降级时是空列表——调用方不用判断开没开。"""
    settings = get_settings()
    return _callbacks_for(
        settings.langfuse_enabled,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )


def _enter_chat_trace(*, conversation_id: int, session_id: str):
    """建根 span + 挂 session_id。任何一步失败返回 None(降级为不上报)。"""
    if not get_langfuse_callbacks():
        return None
    try:
        from langfuse import get_client, propagate_attributes

        settings = get_settings()
        client = get_client(public_key=settings.langfuse_public_key)
        stack = ExitStack()
        span = stack.enter_context(
            client.start_as_current_observation(as_type="span", name=f"chat:{conversation_id}")
        )
        stack.enter_context(propagate_attributes(session_id=_valid_attr(session_id)))
        return stack, span
    except Exception:
        logger.exception("Langfuse trace 根 span 创建失败,本轮不上报")
        return None


@contextmanager
def open_chat_trace(*, conversation_id: int, session_id: str) -> Iterator[Any]:
    """一轮对话的 trace 根 span。关闭时 yield None,调用方原样往下跑。

    astream 是异步生成器,图在迭代时才执行,所以这个上下文必须包住
    `async for` 循环,不能只包函数:

        with open_chat_trace(conversation_id=cid, session_id=sid) as span:
            async for chunk in graph.astream(...):
                ...
    """
    entered = _enter_chat_trace(conversation_id=conversation_id, session_id=session_id)
    if entered is None:
        yield None
        return
    stack, span = entered
    try:
        yield span
    finally:
        stack.close()
        flush_traces()


def record_intent_route(
    span: Any, *, intent: str, route: str, confidence: float | None = None
) -> None:
    """意图/路由补写 trace 元数据(图跑出来才知道,`intent` 事件到达即写)。

    span 为 None(未启用)时是空操作。
    """
    _patch_trace_attrs(
        span,
        metadata={
            "intent": intent or "",
            "route": route or "",
            "intent_confidence": round(float(confidence or 0.0), 3),
        },
    )


def mark_abstain(span: Any) -> None:
    """拒答轮打 `abstain` tag,UI 里按 tag 过滤。span 为 None 时是空操作。"""
    _patch_trace_attrs(span, tags=["abstain"])


def _patch_trace_attrs(span: Any, *, metadata: dict | None = None, tags: list | None = None) -> None:
    """往根 span 补写 trace 级属性(metadata / tags)。

    `use_span` 把根 span 设为当前 span,propagate_attributes 才会把属性
    写到根上;不包这一层的话,属性会落到此刻恰好活跃的那个 span(可能是
    某个节点 span),trace 元数据面板就丢了。
    """
    if span is None:
        return
    try:
        from opentelemetry import trace as otel_trace
        from langfuse import propagate_attributes

        otel_span = getattr(span, "_otel_span", None)
        if otel_span is None or not otel_span.is_recording():
            return
        with otel_trace.use_span(otel_span), propagate_attributes(metadata=metadata, tags=tags):
            pass
    except Exception:
        logger.debug("trace 属性补写失败(不影响对话)", exc_info=True)


def flush_traces() -> None:
    """把当前批次 trace 推出去。

    SDK 默认异步批量(512 条或 5 秒一波),短生命周期场景要即时可见就得
    显式 flush。失败只记日志——观测不能拖慢或搞崩聊天。
    """
    if not get_langfuse_callbacks():
        return
    try:
        from langfuse import get_client

        settings = get_settings()
        get_client(public_key=settings.langfuse_public_key).flush()
    except Exception:
        logger.debug("Langfuse flush 失败(不影响对话)", exc_info=True)
