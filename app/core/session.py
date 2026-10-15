"""In-memory session store with token-budget history trimming."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.config import get_settings


class SessionStore:
    """Per-process chat history keyed by session_id. Lost on restart by design."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}
        self.token_budget: int = get_settings().history_token_budget

    def append_user(self, session_id: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(HumanMessage(content=content))

    def append_assistant(self, session_id: str, content: str) -> None:
        self._sessions.setdefault(session_id, []).append(AIMessage(content=content))

    def get_history(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def trim_history(self, messages: list[BaseMessage]) -> list[BaseMessage]:
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
            max_tokens=self.token_budget,
            start_on="human",
            end_on=("human", "ai"),
            allow_partial=False,
        )
        return system + list(trimmed)
