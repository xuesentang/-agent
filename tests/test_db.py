import pytest
from sqlalchemy.orm import sessionmaker

from customer_service.db import make_engine
from customer_service.models import Base
from customer_service.repository import MessageRow, Repository
from customer_service.seed import init_database


def test_four_tables_seed_and_conversation_history(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    assert set(Base.metadata.tables) == {"faq", "conversations", "messages", "tickets"}
    init_database(engine)
    init_database(engine)
    repo = Repository(sessionmaker(engine, expire_on_commit=False))
    cid = repo.create_conversation()
    assert repo.get_conversation(cid).user_id == "guest"
    assert len(repo.search_faq("退货政策")) == 1
    assert repo.search_faq("邮费") == []
    repo.append_messages(cid, [
        MessageRow("user", "订单 1001"),
        MessageRow("assistant", "", [{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}]),
        MessageRow("tool", '{"status":"shipped"}', tool_call_id="c1"),
    ])
    rows = repo.list_messages(cid)
    assert [row.role for row in rows] == ["user", "assistant", "tool"]
    assert rows[1].tool_calls[0]["id"] == rows[2].tool_call_id
    ticket = repo.create_ticket(cid, "请转人工", "complaint")
    assert ticket.conversation_id == cid
    assert ticket.ticket_id != repo.create_ticket(cid, "再次申请", "complaint").ticket_id


def test_mysql_seed_refuses_non_project_database():
    engine = make_engine("mysql+pymysql://example:example@127.0.0.1:3306/other_database")
    try:
        with pytest.raises(ValueError, match="customer_service"):
            init_database(engine)
    finally:
        engine.dispose()
