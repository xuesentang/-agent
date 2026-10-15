from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.prompts import (
    CUSTOMER_SERVICE_SYSTEM_PROMPT,
    build_chat_prompt,
    build_extract_prompt,
)


def test_chat_prompt_structure():
    messages = build_chat_prompt().format_messages(messages=[], input="在吗")
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "在吗"


def test_chat_prompt_injects_history():
    messages = build_chat_prompt().format_messages(
        messages=[HumanMessage(content="上一轮"), AIMessage(content="好的")],
        input="这一轮",
    )
    assert messages[1].content == "上一轮"
    assert messages[2].content == "好的"
    assert messages[-1].content == "这一轮"


def test_extract_prompt_mentions_json():
    text = build_extract_prompt().format(text="鞋子开胶了")
    assert "json" in text.lower()


def test_system_prompt_has_behavior_constraints():
    prompt = CUSTOMER_SERVICE_SYSTEM_PROMPT
    for keyword in ("退款", "人工", "不知道"):
        assert keyword in prompt
