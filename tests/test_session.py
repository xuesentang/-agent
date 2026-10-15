from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.session import SessionStore


def test_unknown_session_starts_empty():
    assert SessionStore().get_history("nope") == []


def test_append_and_read_back():
    store = SessionStore()
    store.append_user("s1", "你好")
    store.append_assistant("s1", "您好，请问有什么可以帮您？")
    history = store.get_history("s1")
    assert len(history) == 2
    assert history[0].content == "你好"
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)


def test_trim_no_dangling_human():
    store = SessionStore()
    store.token_budget = 55  # 单条 34/21 token，55 恰好保留末尾一轮
    messages = [HumanMessage(content="很长的第一句" * 20), AIMessage(content="回复一" * 20),
                HumanMessage(content="很长的第二句" * 20), AIMessage(content="回复二" * 20)]
    trimmed = store.trim_history(messages)
    assert trimmed[-1].type in ("human", "ai")
    assert len(trimmed) < len(messages)


def test_trim_cannot_fit_any_message_returns_empty():
    store = SessionStore()
    store.token_budget = 10  # 单条 34 token > 10，无完整消息可保留
    messages = [HumanMessage(content="很长的第一句" * 20), AIMessage(content="回复一" * 20)]
    assert store.trim_history(messages) == []


def test_trim_keeps_system_first():
    store = SessionStore()
    store.token_budget = 30
    system = SystemMessage(content="你是电商售后客服")
    messages = [system] + [HumanMessage(content="历史消息" * 30), AIMessage(content="回复" * 30)] * 3
    trimmed = store.trim_history(messages)
    assert trimmed[0] is system
    assert len(trimmed) < len(messages)


def test_trim_within_budget_keeps_all():
    store = SessionStore()
    messages = [HumanMessage(content="短消息"), AIMessage(content="短回复")]
    assert store.trim_history(messages) == messages
