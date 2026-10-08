import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from customer_service.history import ConversationStore, fit_history
from customer_service.prompts import CHAT_PROMPT


def test_store_keeps_complete_turns():
    store = ConversationStore()
    conversation_id = store.create()
    assert store.get(conversation_id) == []
    store.append_turn(conversation_id, "订单 A", "收到 A")
    store.append_turn(conversation_id, "再问", "答复")
    assert [type(m) for m in store.get(conversation_id)] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    with pytest.raises(KeyError):
        store.get("missing")


def test_fit_history_drops_oldest_whole_turn():
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧答复"), HumanMessage(content="新问题"), AIMessage(content="新答复")]
    newest_count = count_tokens_approximately(CHAT_PROMPT.format_messages(history=history[-2:], message="当前问题"))
    result = fit_history(history, "当前问题", budget=newest_count, output_reserve=0)
    assert result == history[-2:]


def test_current_message_over_budget():
    with pytest.raises(ValueError, match="budget"):
        fit_history([], "很长" * 1000, budget=10)


def test_output_tokens_are_reserved():
    current_count = count_tokens_approximately(CHAT_PROMPT.format_messages(history=[], message="你好"))
    with pytest.raises(ValueError, match="budget"):
        fit_history([], "你好", budget=current_count + 511)


def test_fit_history_keeps_tool_call_and_result_together():
    turn = [
        HumanMessage(content="订单 1001 的物流到哪了"),
        AIMessage(content="", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}]),
        ToolMessage(content="运输中", tool_call_id="call_1"),
        AIMessage(content="目前运输中"),
    ]
    result = fit_history(turn, "再问", budget=4000)
    assert result == turn
