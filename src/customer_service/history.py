from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from customer_service.prompts import CHAT_PROMPT


class ConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, list[BaseMessage]] = {}

    def create(self) -> str:
        conversation_id = uuid4().hex
        self._items[conversation_id] = []
        return conversation_id

    def get(self, conversation_id: str) -> list[BaseMessage]:
        return list(self._items[conversation_id])

    def append_turn(self, conversation_id: str, user: str, assistant: str) -> None:
        self._items[conversation_id].extend([HumanMessage(content=user), AIMessage(content=assistant)])


def fit_history(history: list[BaseMessage], current_message: str, budget: int, output_reserve: int = 512) -> list[BaseMessage]:
    input_budget = budget - output_reserve
    def count(candidate: list[BaseMessage]) -> int:
        return count_tokens_approximately(CHAT_PROMPT.format_messages(history=candidate, message=current_message))

    if count([]) > input_budget:
        raise ValueError("Current message exceeds token budget")
    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for item in history:
        if isinstance(item, HumanMessage):
            if current:
                turns.append(current)
            current = [item]
        elif current:
            current.append(item)
    if current:
        turns.append(current)
    selected: list[BaseMessage] = []
    for turn in reversed(turns):
        if not isinstance(turn[-1], AIMessage) or turn[-1].tool_calls:
            continue
        candidate = turn + selected
        if count(candidate) > input_budget:
            break
        selected = candidate
    return selected
