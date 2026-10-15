"""History trimming utilities (CH01 session.py 演化而来)。

CH02 起会话历史落 MySQL messages 表,进程内存 SessionStore 退役;
token 预算裁剪逻辑(纯函数)保留复用,trim 的对象从内存 list 变为
从消息表重建的 LangChain 消息列表。
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


def trim_history(
    messages: list[BaseMessage], token_budget: int
) -> list[BaseMessage]:
    """Trim conversation history to the token budget.

    A leading SystemMessage, if present, is exempt from the budget and kept
    first; the budget applies to the remaining history only.
    """
    system: list[BaseMessage] = []
    history = messages
    if messages and isinstance(messages[0], SystemMessage):
        system, history = [messages[0]], messages[1:]

    trimmed = trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=token_budget,
        start_on="human",
        end_on=("human", "ai"),
        allow_partial=False,
    )
    return system + list(trimmed)
