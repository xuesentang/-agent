import json
from random import Random

from sqlalchemy.orm import sessionmaker

from customer_service.business_tools import build_business_tools
from customer_service.db import make_engine
from customer_service.repository import Repository
from customer_service.seed import init_database


def test_five_tools_query_and_ticket(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'tools.db'}")
    init_database(engine)
    repo = Repository(sessionmaker(engine, expire_on_commit=False))
    cid = repo.create_conversation()
    tools = {tool.name: tool for tool in build_business_tools(repo, cid, Random(7))}
    assert set(tools) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    for name, args, key, value in [
        ("query_order", {"order_id": "1001"}, "order_id", "1001"),
        ("query_product", {"product_name": "耳机"}, "product_name", "耳机"),
        ("query_logistics", {"order_id": "1001"}, "order_id", "1001"),
    ]:
        result = json.loads(tools[name].invoke(args))
        assert result["source"] == "demo"
        assert result[key] == value
    assert json.loads(tools["query_faq"].invoke({"keyword": "退货政策"}))["found"] is True
    assert json.loads(tools["query_faq"].invoke({"keyword": "邮费"}))["found"] is False
    ticket = json.loads(tools["create_ticket"].invoke({"description": "请人工处理", "ticket_type": "complaint"}))
    assert ticket["conversation_id"] == cid
    assert ticket["ticket_id"].startswith("TK-")
