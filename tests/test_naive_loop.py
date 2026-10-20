"""裸 Agent 循环的单测:四条停止条件各一条 —— 循环最容易死在没有出口上。"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.agent.naive_loop import run_naive_agent
from app.core.tools.registry import ToolRegistry


class ScriptedModel:
    """按脚本逐步返回:第 n 次调用返回第 n 个 AIMessage,用尽后重复最后一个。"""

    def __init__(self, script: list[AIMessage]):
        self._script = script
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        msg = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return msg


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return f"订单 {order_id}:轻弹缓震跑步鞋 × 1,状态「已发货」"


@tool
def fake_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(fake_order)
    reg.register(fake_logistics)
    return reg


def _call(name: str, cid: str) -> dict:
    return {"name": name, "args": {"order_id": "1001"}, "id": cid, "type": "tool_call"}


async def test_no_tool_call_converges_in_one_step():
    model = ScriptedModel([AIMessage(content="您好,请问有什么可以帮您?")])
    run = await run_naive_agent("你好", model=model, registry=_registry())
    assert run.stopped_by == "converged"
    assert run.model_calls == 1
    assert run.steps == []
    assert run.reply.startswith("您好")


async def test_tool_call_is_executed_and_fed_back():
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_call("fake_order", "c1")]),
            AIMessage(content="您的订单已发货。"),
        ]
    )
    run = await run_naive_agent("订单1001", model=model, registry=_registry())
    assert run.stopped_by == "converged"
    assert run.model_calls == 2
    assert [s.tool for s in run.steps] == ["fake_order"]
    # 第二次调用必须看得见工具结果,否则「回灌」就是假的
    fed_back = model.seen[1][-1]
    assert "已发货" in str(fed_back.content)


async def test_two_tools_in_sequence_take_three_model_calls():
    """验收标准 5 的最小原型:先查单再查物流,ReAct 走了不止一步。"""
    model = ScriptedModel(
        [
            AIMessage(content="", tool_calls=[_call("fake_order", "c1")]),
            AIMessage(content="", tool_calls=[_call("fake_logistics", "c2")]),
            AIMessage(content="已到北京转运中心。"),
        ]
    )
    run = await run_naive_agent("先查订单1001再查物流", model=model, registry=_registry())
    assert [s.tool for s in run.steps] == ["fake_order", "fake_logistics"]
    assert run.model_calls == 3


async def test_max_steps_stops_the_loop():
    """模型一直要工具:不给它无限次。"""
    model = ScriptedModel([AIMessage(content="", tool_calls=[_call("fake_order", "c1")])])
    run = await run_naive_agent("死循环", model=model, registry=_registry(), max_steps=3)
    assert run.stopped_by == "max_steps"
    assert len(run.steps) == 3
    assert "轻弹缓震跑步鞋" in run.reply  # 撞上限也要把查到的东西交出去


async def test_token_budget_stops_the_loop():
    model = ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[_call("fake_order", "c1")],
                usage_metadata={"input_tokens": 800, "output_tokens": 400, "total_tokens": 1200},
            )
        ]
    )
    run = await run_naive_agent("预算见底", model=model, registry=_registry(), token_budget=1000)
    assert run.stopped_by == "token_budget"
    assert run.model_calls == 1
    assert run.tokens == 1200
