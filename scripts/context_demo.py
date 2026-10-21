"""第七章验收脚本:连聊 N 轮,看预算是否撑得住、降级与摘要有没有按规矩发生。

两种跑法:

    # ① 默认窗口下聊 20 轮:不该触发任何降级和摘要(验收 1、3)
    python scripts/context_demo.py --turns 20 --expect-cascade off

    # ② 演示配置下聊到级联发生(验收 2、4):窗口压到 18000,层 2 很快攒满
    python scripts/context_demo.py --turns 14 --demo-config --expect-cascade on

脚本自己起库、自己装配依赖(真模型 + 真 Milvus),不用起服务;日志直接打屏,
所以 `layer1 降级` / `summary trigger` / `summary done` / `model_ctx` / `history_ctx`
都能在输出里 grep 到。

为什么不走 HTTP:这一章要看的是「每轮实际发出去什么」,服务端日志最好和轮次一一对应;
套一层 curl + 后台进程只会让 grep 更难。聊天接口本身另有验收(README CH07 段)。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 演示配置必须在 import app.config 之前进环境(settings 带 lru_cache)
DEMO_ENV = {
    "MODEL_CONTEXT_WINDOW": "18000",
    "MAX_OUTPUT_TOKENS": "2000",
    "MAX_USER_INPUT_TOKENS": "2000",
    "AGENT_MAX_STEPS": "3",
    "TOOL_RESULT_MAX_TOKENS": "1200",
    "RERANK_TOP_K": "5",
}

QUESTIONS = [
    "邮费是多少",
    "订单 1001 的物流到哪了",
    "我订单 1003 买的是什么",
    "这个能退吗",
    "你好呀",
    "七天无理由怎么算",
    "订单 1005 花了多少钱",
    "退货运费谁出",
    "我要投诉",
    "W40 有宽楦版本吗",
    "订单 1001 现在什么状态",
    "质保多久",
    "那 1003 呢",
    "发票怎么开",
    "换货流程是什么",
    "最开始那个订单后来怎么说",
    "订单 1007 到哪了",
    "修改地址可以吗",
    "退货要多久到账",
    "谢谢",
]


class CountHandler(logging.Handler):
    """数三类事件,顺便留最后一条摘要内容。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.demotions = 0
        self.summary_triggers = 0
        self.summary_done = 0
        self.summary_fail = 0
        self.model_ctx = 0
        self.history_ctx = 0
        self.last_summary = ""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "layer1 降级" in msg:
            self.demotions += 1
        if "summary trigger" in msg:
            self.summary_triggers += 1
        if "summary done" in msg:
            self.summary_done += 1
        if "summary fail" in msg:
            self.summary_fail += 1
        if "model_ctx" in msg:
            self.model_ctx += 1
        if "history_ctx" in msg:
            self.history_ctx += 1


async def run_demo(turns: int, *, demo_config: bool) -> dict:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.core.token_budget import log_startup_check
    from app.logging_setup import configure_logging
    from app.models import Base
    from app.repositories import conversation_repo
    from app.services.agent_chat import build_default_deps, chat_with_agent

    os.environ.setdefault("LOG_LEVEL", "INFO")
    configure_logging()
    handler = CountHandler()
    logging.getLogger().addHandler(handler)

    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    budget = log_startup_check()
    print(f"\n=== 预算 ===\n{budget.as_log()}\n")
    print(f"=== 配置 === demo={demo_config} window={budget.window} turns={turns} ===\n")

    deps = build_default_deps(maker)
    session_id = f"ctx-demo-{os.getpid()}"
    conversation_id = None
    failures = 0

    for i in range(turns):
        question = QUESTIONS[i % len(QUESTIONS)]
        async with maker() as session:
            events = [
                e
                async for e in chat_with_agent(
                    session_id,
                    question,
                    conversation_id,
                    new_conversation=(i == 0),
                    session=session,
                    deps=deps,
                )
            ]
        kinds = [e.event for e in events]
        done = next((e.data for e in events if e.event == "done"), {})
        conversation_id = done.get("conversation_id") or conversation_id
        err = next((e.data for e in events if e.event == "error"), None)
        if err:
            failures += 1
        print(
            f"[{i + 1:02d}] {question[:16]:18s} -> route={done.get('route')} "
            f"finish={done.get('finish_reason')} events={len(kinds)}"
            + (f" ERROR={err.get('code')}" if err else "")
        )

    # 后台摘要可能还在跑:等它们收尾(生产代码不等,测试要等)
    await asyncio.sleep(6)

    async with maker() as session:
        conv = await session.get(conversation_repo.Conversation, conversation_id)
        anchors = (conv.summary_upto_msg_id, conv.layer1_from_msg_id) if conv else (None, None)
    await engine.dispose()

    result = {
        "turns": turns,
        "failures": failures,
        "demotions": handler.demotions,
        "summary_triggers": handler.summary_triggers,
        "summary_done": handler.summary_done,
        "summary_fail": handler.summary_fail,
        "model_ctx": handler.model_ctx,
        "history_ctx": handler.history_ctx,
        "anchors": anchors,
        "budget": budget.as_log(),
    }
    print("\n=== 结果 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--demo-config", action="store_true", help="用验收 2 那组小窗口配置")
    parser.add_argument(
        "--expect-cascade",
        choices=["on", "off"],
        default="off",
        help="on = 期望出现降级与摘要;off = 期望一次都不出现(装得下就不压)",
    )
    args = parser.parse_args()

    if args.demo_config:
        os.environ.update(DEMO_ENV)

    result = asyncio.run(run_demo(args.turns, demo_config=args.demo_config))

    problems: list[str] = []
    if result["failures"]:
        problems.append(f"{result['failures']} 轮出现 error 事件(不该崩)")
    if result["history_ctx"] < result["turns"]:
        problems.append(f"history_ctx 只打了 {result['history_ctx']} 次,应每轮一次")
    if args.expect_cascade == "on":
        if not result["demotions"]:
            problems.append("期望出现 layer1 降级,实际一次都没有")
        if not result["summary_done"]:
            problems.append("期望出现 summary done,实际没有摘要落库")
    else:
        if result["demotions"] or result["summary_triggers"]:
            problems.append(
                f"默认配置下不该压:降级 {result['demotions']} 次、摘要触发 {result['summary_triggers']} 次"
            )

    print()
    if problems:
        for p in problems:
            print(f"FAIL {p}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
