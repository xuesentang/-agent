"""按意图的 token 统计（ch09 需求 6 的 cost 那一条）。

## 为什么是估算，不是账单

真 token 用量得从模型响应里的 `usage_metadata` 拿，但那一层被 ReAct 循环吃掉了：
LangGraph 的 custom 事件里只有文本，usage 没往外传。要采真实值就得改 CH02 的核心
链路（`react.py` 里累积 usage 再塞回 state），本章不动它。

所以这里用 `estimate_tokens` 估：**同比例放大，能比「哪类意图烧钱」，不能当账单。**
每行都写 `token_source="estimated"`，接口返回时也带出来，免得有人拿着报表去对账。

## 为什么记在 agent_chat 里而不是中间件

统计的维度是「意图」，而意图只在图跑完之后才知道。中间件拿不到，
硬要拿就得再解析一遍 SSE 流。
"""

from __future__ import annotations

import logging

from app.repositories import eval_run_repo

logger = logging.getLogger(__name__)


def estimate_text_tokens(text) -> int:
    from app.core.token_budget import estimate_tokens

    return estimate_tokens(str(text or ""))


def estimate_turn(state: dict, *, message: str = "") -> tuple[int, int]:
    """一轮的 (prompt, completion) 估算。

    prompt 侧按层拼：系统提示 + 工具定义 + 层 1（本轮问题）+ 层 2（历史）。
    跟上下文预算算的是同一批东西，只是这里不设上限、照实累加。
    """
    from app.core.prompts import CUSTOMER_SERVICE_TOOLS_SYSTEM_PROMPT
    from app.core.token_budget import tool_schema_tokens

    prompt = estimate_text_tokens(CUSTOMER_SERVICE_TOOLS_SYSTEM_PROMPT)
    prompt += tool_schema_tokens()
    prompt += estimate_text_tokens(state.get("resolved_query") or state.get("canonical_query") or message)
    for item in state.get("history") or []:
        prompt += estimate_text_tokens(getattr(item, "content", item))
    prompt += estimate_text_tokens(state.get("summary") or "")
    # 工具结果与轨迹也要进 prompt（下一轮 Agent 看得到），照实加
    for item in state.get("trail") or []:
        prompt += estimate_text_tokens(getattr(item, "content", ""))

    completion = estimate_text_tokens(state.get("reply") or "")
    return prompt, completion


async def record_turn(
    session,
    *,
    intent: str | None,
    route: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    conversation_id: int | None = None,
    model: str | None = None,
) -> None:
    """写一行。失败只记日志：统计是旁路，不该因为它把一轮对话搞崩。"""
    if not intent:
        return
    try:
        await eval_run_repo.add_usage(
            session,
            intent=intent,
            route=route,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_source="estimated",
            conversation_id=conversation_id,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - 统计失败不影响主链路
        logger.warning("token 统计写失败:%s", exc)


async def by_intent(session) -> dict:
    """按意图聚合 + 总量。占比口径是「占全部 token 的百分比」。"""
    items = await eval_run_repo.usage_by_intent(session)
    total = sum(item["total_tokens"] for item in items)
    turns = sum(item["turns"] for item in items)
    return {
        "total_tokens": total,
        "turns": turns,
        "tokens_per_turn": round(total / turns, 1) if turns else 0.0,
        "items": items,
        "empty": not items,
        "token_source": "estimated",
        "note": "估算值（系统提示 + 工具定义 + 上下文 + 答复），同比例放大，可比较、不当账单",
    }
