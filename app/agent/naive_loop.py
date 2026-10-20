"""裸手写的 Agent 循环(先看清它是什么,再上图)。

剥掉所有框架之后,一个 Agent 就是这段 while:
  调模型 → 它要工具就执行 → 把工具结果塞回消息列表 → 再调 → 直到它不再要工具。

三个停止条件,缺一个就会出事:
1. 模型不再请求工具(正常收敛);
2. 步数上限(模型来回要工具,不给它无限次);
3. token 预算(每步都在烧钱,预算见底必须收).

这个文件在 LangGraph 版本落地后仍然保留:同一件事的两种写法摆在一起,
才看得出「图」到底替我们管了什么(状态、分支、持久化、可观测),又没替我们管什么
(推理本身还是循环里那几次模型调用)。

命令行跑一条看看:
    python -m app.agent.naive_loop "订单 1001 的物流到哪了"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4
DEFAULT_TOKEN_BUDGET = 6000
MAX_TOOL_CONTENT = 1500

SYSTEM_PROMPT = """你是电商平台客服"小智"。需要真实数据时调用工具,不要凭记忆作答。
查到数据就基于数据回答;工具说没查到就如实说没查到。回复控制在 200 字以内。"""


@dataclass
class Step:
    """一步工具调用:给 CLI 与单测看的轨迹,不是给模型看的。"""

    index: int
    tool: str
    args: dict
    ok: bool
    content: str


@dataclass
class NaiveRun:
    reply: str
    steps: list[Step] = field(default_factory=list)
    model_calls: int = 0
    tokens: int = 0
    stopped_by: str = "converged"  # converged | max_steps | token_budget

    def summary(self) -> str:
        lines = [f"[{self.stopped_by}] {self.model_calls} 次模型调用 / 约 {self.tokens} token"]
        lines += [f"  {s.index}. {s.tool}({s.args}) -> {'ok' if s.ok else 'fail'}" for s in self.steps]
        lines.append(f"reply: {self.reply}")
        return "\n".join(lines)


def _count_tokens(message) -> int:
    """优先用上游回传的 usage;拿不到就按字数粗估(中文约 1 字 1 token)。"""
    usage = getattr(message, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total:
        return int(total)
    return len(str(getattr(message, "content", "") or ""))


def _fallback_reply(steps: list[Step]) -> str:
    """撞上限时不再调模型:把已经查到的东西老实交出去,比硬编一句话强。"""
    if not steps:
        return "这个问题我暂时处理不了,您可以补充一下具体订单号或问题细节,我再帮您看看。"
    body = ";".join(s.content[:200] for s in steps if s.ok)
    return f"我查到了这些信息:{body}。如需进一步处理,可以转人工或建工单。"


async def run_naive_agent(
    question: str,
    *,
    model,
    registry: ToolRegistry,
    executor: ToolExecutor | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> NaiveRun:
    """最裸的 Agent:一个 for 循环装下全部逻辑。"""
    executor = executor or ToolExecutor(registry)
    bound = model.bind_tools(registry.for_bind())
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    run = NaiveRun(reply="")
    for step_no in range(1, max_steps + 1):
        ai = await bound.ainvoke(messages)
        run.model_calls += 1
        run.tokens += _count_tokens(ai)

        if not ai.tool_calls:
            run.reply = str(ai.content or "").strip()
            run.stopped_by = "converged"
            return run

        messages.append(ai)
        for call in ai.tool_calls:
            result = await executor.execute(call)
            run.steps.append(
                Step(step_no, result.name, dict(call.get("args") or {}), result.ok, result.content)
            )
            messages.append(ToolMessage(content=result.content, tool_call_id=result.tool_call_id))

        if run.tokens >= token_budget:
            run.stopped_by = "token_budget"
            run.reply = _fallback_reply(run.steps)
            return run

    run.stopped_by = "max_steps"
    run.reply = _fallback_reply(run.steps)
    return run


def main(argv: list[str] | None = None) -> int:
    import sys

    from app.core.llm import get_chat_model
    from app.core.tools.business_tools import build_default_registry

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('用法: python -m app.agent.naive_loop "订单 1001 的物流到哪了"')
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run = asyncio.run(
        run_naive_agent(
            " ".join(args),
            model=get_chat_model(),
            registry=build_default_registry(),
        )
    )
    print(run.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
