import json
from random import Random

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from customer_service.repository import Repository


class OrderInput(BaseModel):
    order_id: str = Field(description="要查询的订单号")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称")


class FAQInput(BaseModel):
    keyword: str = Field(description="用户关于政策、配送或售后的完整问题；支持语义检索。")


class TicketInput(BaseModel):
    description: str = Field(description="用户要求人工处理的问题描述")
    ticket_type: str = Field(description="工单类型，例如 complaint、refund、logistics 或 other")


def build_business_tools(repository: Repository, conversation_id: str, rng: Random, knowledge_search=None) -> list[BaseTool]:
    @tool(args_schema=OrderInput)
    def query_order(order_id: str) -> str:
        """查询订单演示状态。用户询问订单当前状态时使用；返回的是模拟数据。"""
        return json.dumps({"source": "demo", "order_id": order_id, "status": rng.choice(["paid", "shipped", "delivered"]), "note": "演示订单数据"}, ensure_ascii=False)

    @tool(args_schema=ProductInput)
    def query_product(product_name: str) -> str:
        """查询商品演示信息。用户询问商品价格或库存时使用；返回的是模拟数据。"""
        return json.dumps({"source": "demo", "product_name": product_name, "price_yuan": rng.randint(39, 499), "in_stock": rng.choice([True, False])}, ensure_ascii=False)

    @tool(args_schema=OrderInput)
    def query_logistics(order_id: str) -> str:
        """查询订单物流演示节点。用户问物流到哪了、运输进度时使用；返回的是模拟数据。"""
        return json.dumps({"source": "demo", "order_id": order_id, "status": rng.choice(["已揽收", "运输中", "派送中"]), "location": rng.choice(["长沙转运中心", "岳麓配送站"]), "note": "演示物流数据"}, ensure_ascii=False)

    @tool(args_schema=FAQInput)
    def query_faq(keyword: str) -> str:
        """按用户问题语义检索知识库中的政策、配送及售后说明。"""
        if knowledge_search is None:
            raise RuntimeError("Knowledge search is not configured")
        rows = knowledge_search.search(keyword)
        return json.dumps({"found": bool(rows), "matches": [vars(row) for row in rows]}, ensure_ascii=False)

    @tool(args_schema=TicketInput)
    def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单。仅用户明确要求转人工或建工单时调用。"""
        return json.dumps(vars(repository.create_ticket(conversation_id, description, ticket_type)), ensure_ascii=False)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
