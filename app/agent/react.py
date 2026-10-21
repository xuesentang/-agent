"""主力 Agent:一个跑在图节点里的 ReAct 循环。

为什么不用 `create_react_agent`:它把两件本章必须能看见的事藏起来了——**停在哪**
(步数/预算)和**中间发生了什么**(每一步的工具调用)。这两件事恰恰是运维和验收
要的东西,所以循环自己写,只用框架给的 `bind_tools` + `astream`。

三个停止条件,少一个都会出事:
1. 模型不再要工具 → 收敛,已有正文就是答复;
2. 步数上限 → 不再要了,把查到的结果如实交出去;
3. token 预算 → 同上,但连"再组织一次语言"都不做(预算见底就别再花)。

流式:每一步都用 `astream`,正文边收边吐给 writer(服务层转成 SSE token 事件);
如果这一步最后带着 tool_calls,那这段正文是模型的"我这就去查",也照吐——用户
看得见"正在查",比白屏好。工具调用块本身不吐给用户。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from app.core.context import assemble_messages, plan_from_history, render_model_ctx
from app.core.prompts import build_agent_system_prompt
from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_CONTENT = 1500


@dataclass
class ReactStep:
    index: int
    tool: str
    args: dict
    ok: bool
    content: str


@dataclass
class ReactOutcome:
    reply: str
    steps: list[ReactStep] = field(default_factory=list)
    trail: list[BaseMessage] = field(default_factory=list)  # 服务层据此落库工具轨迹
    model_calls: int = 0
    tokens: int = 0
    stopped_by: str = "converged"  # converged | max_steps | token_budget


def _count_tokens(message) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total:
        return int(total)
    return len(str(getattr(message, "content", "") or ""))


def _fallback_reply(steps: list[ReactStep]) -> str:
    if not steps:
        return "抱歉,这个问题我没能查到确定的答案。您可以补充订单号或更具体的信息,我再试一次;也可以转人工由同事跟进。"
    ok = [s for s in steps if s.ok]
    if not ok:
        return "抱歉,我查了几次都没成功(工具返回失败)。建议您稍后再试,或者转人工让同事帮您核实。"
    body = ";".join(s.content[:200] for s in ok)
    return f"我查到的信息是:{body}。如果还需要进一步处理,可以转人工或建工单,我帮您往下走。"


async def run_react(
    question: str,
    *,
    model,
    registry: ToolRegistry,
    executor: ToolExecutor | None = None,
    history: list[BaseMessage] | None = None,
    summary: str = "",
    evidence: str = "",
    layer2_count: int = 0,
    max_steps: int = 4,
    token_budget: int = 6000,
    writer=None,
    confirm_write=None,
) -> ReactOutcome:
    """跑一轮 ReAct。writer 用来把 token / 工具事件推给外部(没有就静默)。

    `confirm_write(tool_call) -> bool` 是写操作的确认闸:返回 True 才执行。
    图里传进来的是「推工单预览卡片给前端,等用户点确认」的实现;不传就等于
    写操作一律拒(单测、脚本默认不会误建工单)。
    """
    executor = executor or ToolExecutor(registry)
    bound = model.bind_tools(registry.for_bind())

    plan = plan_from_history(history, layer2_count=layer2_count)
    messages: list[BaseMessage] = assemble_messages(
        system_prompt=build_agent_system_prompt(max_steps=max_steps),
        plan=plan,
        question=question,
        summary=summary,
        evidence=evidence,
    )
    # 每轮把真正发出去的东西打出来(需求 6)。梗概与证据在后面那份 background 块里,
    # 所以这里除了滑窗还要交代它们的有无——不然排查时会以为模型没见过证据。
    logger.info(render_model_ctx(plan, summary=summary, question=question))
    if evidence:
        logger.info("model_ctx evidence=%s字", len(evidence))

    outcome = ReactOutcome(reply="")
    for step_no in range(1, max_steps + 1):
        aggregate = None
        async for chunk in bound.astream(messages):
            aggregate = chunk if aggregate is None else aggregate + chunk
            text = getattr(chunk, "text", "") or ""
            if text and writer is not None:
                writer({"kind": "token", "text": text})
        if aggregate is None:  # 上游一个块都没吐,别硬编
            outcome.stopped_by = "upstream_empty"
            outcome.reply = _fallback_reply(outcome.steps)
            return outcome

        outcome.model_calls += 1
        outcome.tokens += _count_tokens(aggregate)
        text = str(aggregate.text or "")

        if not aggregate.tool_calls:
            outcome.reply = text.strip()
            outcome.stopped_by = "converged"
            return outcome

        messages.append(AIMessage(content=text, tool_calls=aggregate.tool_calls))
        outcome.trail.append(AIMessage(content=text, tool_calls=aggregate.tool_calls))

        for call in aggregate.tool_calls:
            if writer is not None:
                writer({"kind": "tool_start", "tool_call_id": call["id"], "name": call["name"], "args": call.get("args") or {}})
            # 写操作要用户确认(第八章):确认由调用方注入(图里是 interrupt),
            # 没注入就等于「没人能确认」,一律拒——不许默认放行。
            # 返回值三种:False=拒绝;True=同意、就用这次模型给的参数;dict=同意、
            # 但要以用户看到的那份参数为准(图恢复时会重跑节点,模型可能改口)。
            meta = registry.meta(call["name"]) if registry is not None else None
            confirmed = True
            if meta is not None and meta.is_write:
                decision = await confirm_write(call) if confirm_write is not None else False
                confirmed = decision is not False and decision is not None
                if confirmed and isinstance(decision, dict) and decision:
                    if dict(decision) != dict(call.get("args") or {}):
                        logger.info("写操作参数以用户确认的版本为准(模型重跑后改口)")
                    call = {**call, "args": dict(decision)}
            result = await executor.execute(call, confirmed=confirmed)
            outcome.steps.append(
                ReactStep(step_no, result.name, dict(call.get("args") or {}), result.ok, result.content)
            )
            if writer is not None:
                writer(
                    {
                        "kind": "tool_end",
                        "tool_call_id": result.tool_call_id,
                        "name": result.name,
                        "ok": result.ok,
                        "summary": result.content[:MAX_TOOL_CONTENT],
                    }
                )
            tool_msg = ToolMessage(content=result.content, tool_call_id=result.tool_call_id)
            messages.append(tool_msg)
            outcome.trail.append(tool_msg)

        if outcome.tokens >= token_budget:
            outcome.stopped_by = "token_budget"
            outcome.reply = _fallback_reply(outcome.steps)
            return outcome

    outcome.stopped_by = "max_steps"
    outcome.reply = _fallback_reply(outcome.steps)
    return outcome
