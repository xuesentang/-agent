"""主力 Agent(ReAct 循环)单测:步数、预算、追问、工具事件旁路。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.agent.react import run_react
from app.core.tools.registry import ToolRegistry


class ScriptedStreamModel:
    """按脚本逐次吐 chunk 列表:tool_calls 通过 tool_call_chunks 累加(与真实流一致)。"""

    def __init__(self, script: list[list[AIMessageChunk]]):
        self._script = script
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.seen.append(list(messages))
        script = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        for chunk in script:
            yield chunk


def _text(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


def _tool_call_chunk(name: str, cid: str, args: str = '{"order_id": "1001"}') -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": cid, "index": 0}],
    )


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return f"订单 {order_id}:状态「已发货」"


@tool
def fake_logistics(order_id: str) -> str:
    """查物流。order_id: 订单号"""
    return f"订单 {order_id} 已到达北京转运中心"


@tool
def broken_tool(order_id: str) -> str:
    """坏工具。order_id: 订单号"""
    raise RuntimeError("boom")


def _registry(*extra) -> ToolRegistry:
    reg = ToolRegistry()
    for t in (fake_order, fake_logistics) + extra:
        reg.register(t)
    return reg


async def test_single_call_converges_and_streams_tokens():
    events: list[dict] = []
    model = ScriptedStreamModel([[_text("您的订单"), _text("已发货。")]])
    out = await run_react(
        "订单1001什么状态", model=model, registry=_registry(), writer=events.append
    )
    assert out.stopped_by == "converged"
    assert out.reply == "您的订单已发货。"
    assert [e["text"] for e in events if e["kind"] == "token"] == ["您的订单", "已发货。"]
    assert out.model_calls == 1


async def test_two_step_tool_chain():
    """验收标准 5:先查订单再查物流,ReAct 走了不止一步。"""
    events: list[dict] = []
    model = ScriptedStreamModel(
        [
            [_tool_call_chunk("fake_order", "c1")],
            [_tool_call_chunk("fake_logistics", "c2")],
            [_text("订单已发货,最新轨迹是到达北京转运中心。")],
        ]
    )
    out = await run_react("订单1001的物流到哪了", model=model, registry=_registry(), writer=events.append)
    assert [s.tool for s in out.steps] == ["fake_order", "fake_logistics"]
    assert out.model_calls == 3
    assert out.stopped_by == "converged"
    kinds = [e["kind"] for e in events]
    assert kinds.count("tool_start") == 2
    assert kinds.count("tool_end") == 2
    # 第二个工具调用必须看得见第一个工具的结果
    assert "已发货" in str(model.seen[1][-1].content)


async def test_tool_message_ids_match_for_persistence():
    """落库要按 tool_call_id 对号,trail 里 assistant 与 tool 必须配对。"""
    model = ScriptedStreamModel([[_tool_call_chunk("fake_order", "c1")], [_text("好了")]])
    out = await run_react("订单1001", model=model, registry=_registry())
    assistants = [m for m in out.trail if isinstance(m, AIMessage) and m.tool_calls]
    tools = [m for m in out.trail if getattr(m, "tool_call_id", None)]
    assert assistants and tools
    assert tools[0].tool_call_id == assistants[0].tool_calls[0]["id"]


async def test_max_steps_returns_what_was_found():
    model = ScriptedStreamModel([[_tool_call_chunk("fake_order", "c1")]])
    out = await run_react("死循环", model=model, registry=_registry(), max_steps=2)
    assert out.stopped_by == "max_steps"
    assert len(out.steps) == 2
    assert "已发货" in out.reply


async def test_token_budget_stops_before_extra_calls():
    model = ScriptedStreamModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": "fake_order", "args": '{"order_id":"1001"}', "id": "c1", "index": 0}],
                    usage_metadata={"input_tokens": 900, "output_tokens": 300, "total_tokens": 1200},
                )
            ]
        ]
    )
    out = await run_react("预算见底", model=model, registry=_registry(), token_budget=1000)
    assert out.stopped_by == "token_budget"
    assert out.model_calls == 1


async def test_tool_failure_is_fed_back_not_raised():
    model = ScriptedStreamModel(
        [[_tool_call_chunk("broken_tool", "c1")], [_text("抱歉,查询失败了。")]]
    )
    out = await run_react("查订单", model=model, registry=_registry(broken_tool))
    assert out.steps[0].ok is False
    assert "boom" in out.steps[0].content
    assert out.reply == "抱歉,查询失败了。"


async def test_missing_argument_triggers_validation_feedback_for_followup():
    """缺参数不是崩溃,是要模型追问用户 —— 错误说明必须回到模型手里。"""
    model = ScriptedStreamModel(
        [[_tool_call_chunk("fake_order", "c1", args="{}")], [_text("请问您的订单号是多少?")]]
    )
    out = await run_react("我的快递到哪了", model=model, registry=_registry())
    assert out.steps[0].ok is False
    assert "参数不合法" in out.steps[0].content
    assert "订单号" in out.reply


async def test_evidence_is_attached_to_the_question():
    model = ScriptedStreamModel([[_text("满 99 包邮 [1]")]])
    await run_react(
        "邮费多少", model=model, registry=_registry(), evidence="[1] 章节:运费标准\n    内容:满 99 包邮"
    )
    human = model.seen[0][-1]
    assert "【知识库检索结果】" in str(human.content)
    assert "满 99 包邮" in str(human.content)


async def test_no_writer_does_not_raise():
    """图节点也可能被 ainvoke 直接跑(单测/批量),此时 writer 是空实现。"""
    model = ScriptedStreamModel([[_text("好")]])
    out = await run_react("你好", model=model, registry=_registry())
    assert out.reply == "好"
