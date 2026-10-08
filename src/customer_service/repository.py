from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from customer_service.models import Conversation, FAQ, Message, Ticket


@dataclass(frozen=True)
class ConversationRow:
    id: str
    user_id: str
    status: str


@dataclass(frozen=True)
class MessageRow:
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class FAQRow:
    question: str
    answer: str
    category: str


@dataclass(frozen=True)
class TicketRow:
    ticket_id: str
    conversation_id: str
    description: str
    ticket_type: str
    status: str


class Repository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def create_conversation(self, user_id: str = "guest") -> str:
        cid = uuid4().hex
        with self.session_factory.begin() as session:
            session.add(Conversation(id=cid, user_id=user_id or "guest", status="open"))
        return cid

    def get_conversation(self, conversation_id: str) -> ConversationRow | None:
        with self.session_factory() as session:
            item = session.get(Conversation, conversation_id)
            return ConversationRow(item.id, item.user_id, item.status) if item else None

    def list_messages(self, conversation_id: str) -> list[MessageRow]:
        with self.session_factory() as session:
            items = session.scalars(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at, Message.id)).all()
            return [MessageRow(item.role, item.content, item.tool_calls, item.tool_call_id) for item in items]

    def append_messages(self, conversation_id: str, rows: list[MessageRow]) -> None:
        with self.session_factory.begin() as session:
            session.add_all(Message(conversation_id=conversation_id, role=row.role, content=row.content, tool_calls=row.tool_calls, tool_call_id=row.tool_call_id) for row in rows)

    def search_faq(self, keyword: str) -> list[FAQRow]:
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.session_factory() as session:
            items = session.scalars(select(FAQ).where(FAQ.question.like(f"%{escaped}%", escape="\\"))).all()
            return [FAQRow(item.question, item.answer, item.category) for item in items]

    def create_ticket(self, conversation_id: str, description: str, ticket_type: str) -> TicketRow:
        ticket = TicketRow(f"TK-{uuid4().hex[:16].upper()}", conversation_id, description, ticket_type, "open")
        with self.session_factory.begin() as session:
            session.add(Ticket(**vars(ticket)))
        return ticket
