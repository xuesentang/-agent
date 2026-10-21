"""第八章验收脚本:工具注册中心、执行引擎韧性、审计、工单确认流。

五个子命令,对应需求里的几条验收:

    # 验收 1:新写一个工具,只调 register 就上线,不动任何核心代码
    python scripts/tool_demo.py register

    # 验收 3:注册中心里内置 + MCP 同一份表(需先起两个 MCP Server,见 README)
    python scripts/tool_demo.py registry

    # 验收 6:人为超时 / 抖动,看重试次数与审计;写操作一次都不重试
    python scripts/tool_demo.py resilience

    # 验收 5/7:建工单必须经确认。--decision approve 落库,reject 不落库
    python scripts/tool_demo.py confirm --decision approve
    python scripts/tool_demo.py confirm --decision reject

    # 随便什么时候看一眼最近的审计
    python scripts/tool_demo.py audit

默认落 CH07 那套开发库(`./data/ch07.db` + `./data/ch07_milvus.db`,都要用),可用
`DATABASE_URL` / `VECTOR_STORE_URI` 覆盖。脚本自己 create_all,不用先跑迁移。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 必须在 import app.config 之前进环境(settings 带 lru_cache)
# 用 CH07 那套开发库:建单确认那一轮要过退款子流程的政策检索,库和向量库得是齐的
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/ch07.db")
os.environ.setdefault("VECTOR_STORE_URI", "./data/ch07_milvus.db")


def _line(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


async def _prepare_db():
    from app.db import get_engine

    from app.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


# ------------------------------------------------------------------ 验收 1


async def cmd_register() -> int:
    """只调 register 就能上线一个新工具——核心代码一行不改。"""
    from langchain_core.tools import tool

    from app.core.tools.executor import ToolExecutor
    from app.core.tools.registry import ToolMeta, ToolRegistry

    @tool
    def query_coupon(user_key: str) -> str:
        """查询用户可用优惠券。user_key: 用户标识"""
        return f"{user_key} 可用优惠券 3 张:满 200 减 20、满 500 减 60、包邮券 1 张"

    _line("验收 1 · 运行期注册新工具")
    reg = ToolRegistry()
    reg.register(query_coupon, ToolMeta(name="query_coupon", read_only=True))
    print("注册表:", ", ".join(reg.names()))

    executor = ToolExecutor(reg)
    result = await executor.execute({"name": "query_coupon", "args": {"user_key": "u1"}, "id": "demo-1"})
    print("调用结果:", result.content)

    # 顺带验一条规矩:重名必须报错,不能让两个同名工具打架
    try:
        reg.register(query_coupon, ToolMeta(name="query_coupon"))
        print("!! 重名没报错,不合规")
        return 1
    except Exception as exc:  # noqa: BLE001 - 这里就是要看到它抛
        print(f"重名注册被拦下:{exc}")
    return 0


# ------------------------------------------------------------------ 验收 3


async def cmd_registry() -> int:
    """内置 + MCP 在同一份表里,各自带来源与读写标记。"""
    from app.config import get_settings
    from app.core.tools.business_tools import build_default_registry
    from app.core.tools.mcp_source import get_mcp_source, refresh_registry

    settings = get_settings()
    _line(f"验收 3 · 注册中心(MCP_ENABLED={settings.mcp_enabled})")
    reg = build_default_registry()
    if settings.mcp_enabled:
        added = await refresh_registry(reg, get_mcp_source())
        print(f"从 MCP 拉取到 {len(added)} 个工具:{', '.join(added) or '无'}")
    else:
        print("MCP 未开启:只显示内置工具。开法:MCP_ENABLED=true 并起两个 Server")

    print(f"{'工具名':<24}{'来源':<10}{'读写':<6}{'Server'}")
    for name in reg.names():
        meta = reg.meta(name)
        source = getattr(meta, "source", "builtin")
        server = getattr(meta, "server", None) or "-"
        ro = "只读" if getattr(meta, "read_only", True) else "写"
        print(f"{name:<24}{source:<10}{ro:<6}{server}")
    print(f"\n共 {len(reg.names())} 个工具")
    print("注:Server 侧加了工具、只重启那一个 Server,这边下一轮(TTL 30s)就能拉到,")
    print("    客服服务不用重启——重跑本命令即可看到新工具。")
    return 0


# ------------------------------------------------------------------ 验收 6


async def cmd_resilience() -> int:
    """超时 / 抖动 / 写操作重试规矩,结果与审计一起看。"""
    from langchain_core.tools import tool

    from app.core.tools.executor import ToolExecutor
    from app.core.tools.registry import ToolMeta, ToolRegistry

    await _prepare_db()
    from app.db import get_sessionmaker

    attempts = {"flaky": 0, "slow": 0, "slow_write": 0}

    @tool
    async def flaky(query: str) -> str:
        """偶发抖动的只读工具。query: 任意"""
        attempts["flaky"] += 1
        if attempts["flaky"] == 1:
            raise ConnectionError("连接被重置")
        return f"ok:{query}"

    @tool
    async def slow(query: str) -> str:
        """慢只读工具。query: 任意"""
        attempts["slow"] += 1
        await asyncio.sleep(2)
        return "done"

    @tool
    async def slow_write(name: str) -> str:
        """慢写工具。name: 名字"""
        attempts["slow_write"] += 1
        await asyncio.sleep(2)
        return "written"

    reg = ToolRegistry()
    reg.register(flaky)
    reg.register(slow)
    reg.register(slow_write, ToolMeta(name="slow_write", read_only=False))
    executor = ToolExecutor(
        reg,
        timeout=0.2,
        transient_errors=(ConnectionError,),
        session_factory=get_sessionmaker(),
    )

    _line("验收 6 · 韧性")
    cases = [
        ("只读+抖动一次", {"name": "flaky", "args": {"query": "q"}, "id": "r1"}, True),
        ("只读+超时", {"name": "slow", "args": {"query": "q"}, "id": "r2"}, True),
        ("写操作+超时", {"name": "slow_write", "args": {"name": "x"}, "id": "r3"}, True),
    ]
    for label, call, confirmed in cases:
        result = await executor.execute(call, confirmed=confirmed)
        print(
            f"{label:<14} ok={result.ok!s:<5} status={result.status:<6} "
            f"retry={result.retry_count} {result.duration_ms}ms :: {result.content[:40]}"
        )
    print("实际被调用次数:", attempts)
    print("写操作只许调一次:超时未必没执行,重复提交比失败更糟。")

    await _print_audit(limit=5)
    return 0


# ------------------------------------------------------------------ 审计


async def _print_audit(limit: int = 10) -> None:
    from sqlalchemy import select

    from app.db import get_sessionmaker
    from app.models.tool_audit import ToolAuditLog

    async with get_sessionmaker()() as session:
        rows = list(
            (
                await session.execute(
                    select(ToolAuditLog).order_by(ToolAuditLog.id.desc()).limit(limit)
                )
            )
            .scalars()
            .all()
        )
    _line(f"审计最近 {len(rows)} 条")
    for row in rows:
        print(
            f"#{row.id:<4} {row.tool_name:<16} {row.tool_source:<8} {row.status:<8} "
            f"retry={row.retry_count} conv={row.conversation_id} :: {(row.error_message or row.result_summary or '')[:40]}"
        )


async def cmd_audit() -> int:
    await _prepare_db()
    await _print_audit(limit=20)
    return 0


async def cmd_mcp_call() -> int:
    """验收 4:MCP 物流 Server 接管内置 query_logistics,真调一次并留审计。"""
    from app.config import get_settings
    from app.core.tools.business_tools import build_default_registry
    from app.core.tools.executor import ToolExecutor
    from app.core.tools.mcp_source import get_mcp_source, refresh_registry
    from app.db import get_sessionmaker

    settings = get_settings()
    _line(f"验收 4 · MCP 工具接管内置(MCP_ENABLED={settings.mcp_enabled})")
    if not settings.mcp_enabled:
        print("MCP 没开。先起两个 Server,再用 MCP_ENABLED=true 重跑:")
        print("  python scripts/mcp_logistics_server.py --port 8121")
        print("  python scripts/mcp_aftersale_server.py --port 8122")
        print('  MCP_SERVERS="logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp"')
        return 1

    await _prepare_db()
    reg = build_default_registry()
    print("内置 query_logistics 还在?", "query_logistics" in reg.names(), "(应为 False:被 MCP 接管后下线)")
    await refresh_registry(reg, get_mcp_source())

    executor = ToolExecutor(reg, session_factory=get_sessionmaker())
    result = await executor.execute(
        {"name": "query_logistics_track", "args": {"order_id": "1001"}, "id": "mcp-demo-1"}
    )
    print(f"调用 query_logistics_track ok={result.ok} {result.duration_ms}ms")
    print(result.content)
    await _print_audit(limit=3)
    return 0 if result.ok else 1


# ------------------------------------------------------------------ 验收 5/7


async def cmd_confirm(decision: str, question: str, real_model: bool) -> int:
    """跑一轮建单请求:看 interrupt 卡片,再按 decision 恢复。

    默认用脚本化模型(见下面的 `ScriptedTicketModel`),不是图省事:这条要验的是
    「没确认就不许写」「确认后接着跑」「取消就真不写」,而真模型经常先反问订单号、
    这一轮压根不提建单——验收会退化成「验模型今天心情如何」。脚本化模型照样走
    真实的图、真实的执行引擎、真实的库,只是把「模型决定建单」这一步钉死。
    想看真模型表现就加 `--real-model`。
    """
    from sqlalchemy import select

    from app.agent.graph import AgentDeps, build_agent_graph
    from app.core.llm import get_chat_model, get_extract_model
    from app.core.tools.business_tools import build_default_registry
    from app.db import get_sessionmaker
    from app.models.message import Message
    from app.models.ticket import Ticket
    from app.services.agent_chat import chat_with_agent

    await _prepare_db()
    session_factory = get_sessionmaker()
    registry = build_default_registry()

    if real_model:
        from app.services.agent_chat import build_default_deps

        deps = build_default_deps(session_factory)
    else:
        deps = AgentDeps(
            model=ScriptedTicketModel(),
            registry=registry,
            session_factory=session_factory,
            intent_model=FixedIntentModel("物流"),
            check_model=FixedIntentModel(None),
        )
    graph = build_agent_graph(deps)

    _line(f"验收 5/7 · 工单确认流({'真模型' if real_model else '脚本化模型'})")
    print("用户:", question)
    card = None
    conv_id = None
    async with session_factory() as session:
        async for event in chat_with_agent(
            "ch08-demo", question, None, new_conversation=True, session=session, deps=deps, graph=graph
        ):
            kind = event.event
            data = event.data
            if kind == "interrupt":
                card = data
                print("卡片:", data.get("ticket_type"), "|", data.get("description"))
            elif kind == "token":
                print("答复:", data.get("content"))
            elif kind == "tool_end":
                print(f"工具 {data.get('name')} ok={data.get('ok')} {str(data.get('summary'))[:60]}")
            elif kind in ("done", "error"):
                conv_id = data.get("conversation_id", conv_id)
                print(kind, {k: v for k, v in data.items() if k in ("finish_reason", "interrupted", "code")})

        tickets_before = list(
            (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conv_id)
                )
            )
            .scalars()
            .all()
        )
        print(f"点按钮之前本次会话 tickets {len(tickets_before)} 行(必须为 0)")

        if card is None:
            print("这一轮没出卡片——真模型常先追问订单号,换 --question 或去掉 --real-model。")
            return 1

        _line(f"用户点了「{decision}」")
        async for event in chat_with_agent(
            "ch08-demo",
            "",
            conv_id,
            resume={
                "decision": decision,
                "ticket_type": card.get("ticket_type", ""),
                "description": card.get("description", ""),
            },
            session=session,
            deps=deps,
            graph=graph,
        ):
            if event.event == "token":
                print("答复:", event.data.get("content"))
            elif event.event == "tool_end":
                print(f"工具 {event.data.get('name')} ok={event.data.get('ok')}")
            elif event.event in ("done", "error"):
                print(event.event, event.data.get("finish_reason") or event.data.get("code"))

        tickets = list(
            (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conv_id)
                )
            )
            .scalars()
            .all()
        )
        print(f"本次会话 tickets {len(tickets)} 行:", [(t.ticket_no, t.ticket_type) for t in tickets])
        if tickets:
            print("落库描述:", tickets[0].description[:80])
        msgs = list(
            (
                await session.execute(
                    select(Message).where(Message.conversation_id == conv_id).order_by(Message.id)
                )
            )
            .scalars()
            .all()
        )
        print("消息流水:", [m.role for m in msgs])

    await _print_audit(limit=5)
    ok = (decision == "approve" and len(tickets) == 1) or (decision == "reject" and not tickets)
    print("\n结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


class FixedIntentModel:
    """意图固定(物流 → 走业务 Agent,不碰检索),省掉一次模型调用。"""

    def __init__(self, intent: str | None):
        self.intent = intent

    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        if self.intent is None:
            return AIMessage(content='{"useful": true, "reason": "demo"}')
        return AIMessage(content='{"intent": "%s", "confidence": 0.95}' % self.intent)


class ScriptedTicketModel:
    """前两次都提 create_ticket,第三次收尾。

    恢复时节点会从头重跑,所以第二次必须是同一笔调用——顺带也验证了「模型重跑后
    改了措辞也不会替用户改单」这条:真要验改口,把第二次的 description 换掉即可。
    """

    ARGS = {
        "ticket_type": "售后",
        "description": "订单 1001 物流显示已签收但用户未收到货,请专员核实物流真实状态并联系用户。",
    }

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        import json as _json

        from langchain_core.messages import AIMessageChunk, ToolMessage

        self.calls += 1
        if self.calls <= 2:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "create_ticket",
                        "args": _json.dumps(self.ARGS, ensure_ascii=False),
                        "id": f"demo_call_{self.calls}",
                        "index": 0,
                    }
                ],
            )
            return

        # 收尾话术要跟着工具结果走:被拒了就别说「已提交」,那是最容易糊弄人也最
        # 容易露馅的一句。执行引擎把「未确认,已拒绝」回灌给模型,模型据此改口。
        denied = any(
            isinstance(m, ToolMessage) and "未确认" in str(m.content or "")
            for m in messages
        )
        if denied:
            yield AIMessageChunk(content="好的,没有提交工单。您需要的话随时说,我再帮您建。")
            return
        yield AIMessageChunk(content="工单已提交,专员会在 24 小时内联系您。")


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description="第八章验收脚本")
    parser.add_argument(
        "mode",
        choices=["register", "registry", "resilience", "mcp", "confirm", "audit"],
        help="要验哪一条",
    )
    parser.add_argument("--decision", choices=["approve", "reject"], default="approve")
    parser.add_argument("--question", default="订单 1001 物流显示已签收但我没收到,帮我建个工单让专员核实")
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="用真模型跑(会先追问订单号,不稳;默认用脚本化模型验闸门)",
    )
    args = parser.parse_args()

    if args.mode == "register":
        return asyncio.run(cmd_register())
    if args.mode == "registry":
        return asyncio.run(cmd_registry())
    if args.mode == "resilience":
        return asyncio.run(cmd_resilience())
    if args.mode == "mcp":
        return asyncio.run(cmd_mcp_call())
    if args.mode == "audit":
        return asyncio.run(cmd_audit())
    return asyncio.run(cmd_confirm(args.decision, args.question, args.real_model))


if __name__ == "__main__":
    raise SystemExit(main())
