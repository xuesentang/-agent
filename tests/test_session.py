"""trim_history 模块函数 TDD(CH01 session.py 的裁剪逻辑,SessionStore 退役后保留)。

CH02 起历史落 messages 表,裁剪对象变为从消息表重建的 LangChain 消息列表,
纯函数 trim_history(messages, token_budget) 直接复用。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.session import trim_history


def test_trim_no_dangling_human():
    messages = [HumanMessage(content="很长的第一句" * 20), AIMessage(content="回复一" * 20),
                HumanMessage(content="很长的第二句" * 20), AIMessage(content="回复二" * 20)]
    trimmed = trim_history(messages, token_budget=55)  # 单条 34/21 token,55 恰好保留末尾一轮
    assert trimmed[-1].type in ("human", "ai")
    assert len(trimmed) < len(messages)


def test_trim_cannot_fit_any_message_returns_empty():
    messages = [HumanMessage(content="很长的第一句" * 20), AIMessage(content="回复一" * 20)]
    assert trim_history(messages, token_budget=10) == []  # 单条 34 token > 10


def test_trim_keeps_system_first():
    system = SystemMessage(content="你是电商售后客服")
    messages = [system] + [HumanMessage(content="历史消息" * 30), AIMessage(content="回复" * 30)] * 3
    trimmed = trim_history(messages, token_budget=30)
    assert trimmed[0] is system
    assert len(trimmed) < len(messages)


def test_trim_within_budget_keeps_all():
    messages = [HumanMessage(content="短消息"), AIMessage(content="短回复")]
    assert trim_history(messages, token_budget=2048) == messages


def test_trim_keeps_trailing_tool_message():
    """review C1 双保险:带工具轮次结尾 [user, ai(tool_calls), tool] 不得被
    裁成孤立 tool_calls(end_on 含 tool)。"""
    tool_call = {"name": "query_logistics", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"}
    messages = [
        HumanMessage(content="订单1物流"),
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content="已签收", tool_call_id="c1"),
    ]
    trimmed = trim_history(messages, token_budget=2048)
    assert [m.type for m in trimmed] == ["human", "ai", "tool"]
    assert trimmed[2].tool_call_id == "c1"
