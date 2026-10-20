"""五个业务工具(spec §4)。

query_order / query_product / query_logistics:真实场景该调公司电商/物流系统 API,
演示起见工具内部按入参确定性随机生成(random.Random(seed=业务键)),不接真实接口、
不建表——同入参两次调用结果逐字一致(可复现,Review Focus 6)。
query_faq:ch03 起内部换向量语义检索(Milvus Top-K),入参出参契约与 CH02 一致;
向量库/嵌入故障时降级 CH02 的 LIKE 关键词查询,聊天不中断。
create_ticket:写 tickets 表,conversation_id 经 ContextVar 注入(spec §4)。
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Literal

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from app.core.tools.context import current_conversation_id
from app.core.tools.registry import ToolRegistry
from app.rag.retriever import Retriever, retrieve
from app.repositories import faq_repo, ticket_repo

logger = logging.getLogger("kefu_agent")

# ---------------------------------------------------------------- mock 数据素材
_ORDER_PRODUCTS = [
    ("纯棉宽松圆领T恤", 89.00),
    ("无线蓝牙降噪耳机", 399.00),
    ("轻弹缓震跑步鞋", 329.00),
    ("大容量双肩通勤包", 159.00),
    ("智能运动手表", 699.00),
    ("山羊绒混纺围巾", 259.00),
]
_ORDER_STATUS = ["待发货", "已发货", "运输中", "已签收", "已完成"]
_LOGISTICS_STAGES = [
    "已揽收",
    "运输中",
    "到达分拨中心",
    "离开分拨中心",
    "到达目的城市网点",
    "派送中",
    "已签收",
]
_LOGISTICS_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]

# mock 数据的固定时间基准(review Minor 1):由 seed 派生偏移、不用 today/now,
# 保证同一入参跨天调用也逐字一致(可复现是 AGENTS.md 数据规则的要求)。
_MOCK_EPOCH_DATE = date(2026, 1, 1)
_MOCK_EPOCH_DT = datetime(2026, 6, 1, 8, 0, 0)


@tool
def query_order(order_id: str) -> str:
    """查询订单信息:商品、金额、状态、下单时间。
    order_id: 订单号,如 1001"""
    rng = random.Random(f"order:{order_id}")
    name, price = rng.choice(_ORDER_PRODUCTS)
    qty = rng.randint(1, 3)
    status = rng.choice(_ORDER_STATUS)
    order_date = (_MOCK_EPOCH_DATE + timedelta(days=rng.randint(0, 300))).isoformat()
    return (
        f"订单 {order_id}:{name} × {qty},实付 ¥{price * qty:.2f},"
        f"当前状态「{status}」,下单日期 {order_date}。"
    )


@tool
def query_product(product_name: str) -> str:
    """查询商品信息:售价、库存、规格。
    product_name: 商品名称关键词,如「跑鞋」"""
    rng = random.Random(f"product:{product_name}")
    price = rng.choice([39, 59, 89, 129, 199, 259, 329, 399, 599, 899])
    stock = rng.randint(0, 500)
    spec = rng.choice(["标准版", "升级版", "礼盒装"])
    color = rng.choice(["经典黑", "云雾白", "雾霾蓝", "茱萸粉"])
    return f"商品「{product_name}」:{spec} / {color},售价 ¥{price:.2f},库存 {stock} 件。"


@tool
def query_logistics(order_id: str) -> str:
    """查询订单的物流轨迹(最新在上)。
    order_id: 订单号,如 1001"""
    rng = random.Random(f"logistics:{order_id}")
    n = rng.randint(3, 6)
    # 物流按流程顺序推进:取阶段序列的一个连续子段,再倒序时间。
    start = rng.randint(0, len(_LOGISTICS_STAGES) - n)
    stages = _LOGISTICS_STAGES[start : start + n]
    points = []
    t = _MOCK_EPOCH_DT + timedelta(hours=rng.randint(2, 240))
    for stage in reversed(stages):
        points.append((t, rng.choice(_LOGISTICS_CITIES), stage))
        t -= timedelta(hours=rng.randint(4, 30))
    lines = [f"订单 {order_id} 的物流轨迹(最新在上):"]
    lines += [
        f"- {t:%m-%d %H:%M} {city} {stage}" for t, city, stage in points
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- DB 工具(工厂注入 session)
def _format_qa_hits(pairs: list[tuple[str, str]]) -> str:
    """问答命中列表的统一格式(与 CH02 逐字一致,向量/LIKE 两条路共用)。"""
    lines = [f"找到 {len(pairs)} 条相关 FAQ:"]
    lines += [f"{i}. 问:{q}\n   答:{a}" for i, (q, a) in enumerate(pairs, 1)]
    return "\n".join(lines)


def build_db_tools(session_factory, retriever: Retriever | None = None) -> list[BaseTool]:
    """query_faq / create_ticket:需要数据库会话,由工厂注入,不进工具 schema。

    retriever 可注入(单测替身);None 时用进程内真检索器(Milvus + BGE-M3)。
    """

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题(退货政策、运费、发票等客服知识)。
        keyword: 问题关键词,如「退货」「运费」「发票」"""
        try:
            hits = await retrieve(
                session_factory=session_factory, query=keyword, retriever=retriever
            )
        except Exception as exc:  # noqa: BLE001 - 向量库/嵌入故障降级 LIKE,不中断聊天
            logger.warning("query_faq 向量检索失败,降级 LIKE 关键词查询:%s", exc)
            hits = None
        if hits is not None:
            # 向量路:空结果是语义检索的诚实答案,不降级(降级只留给真故障)
            if not hits:
                return f"未在 FAQ 知识库中找到与「{keyword}」相关的内容。"
            return _format_qa_hits([(h.chunk.questions, h.chunk.answer) for h in hits])
        async with session_factory() as session:
            faq_hits = await faq_repo.search_by_keyword(session, keyword)
        if not faq_hits:
            return f"未在 FAQ 知识库中找到与「{keyword}」相关的内容。"
        return _format_qa_hits([(h.question, h.answer) for h in faq_hits])

    @tool
    async def create_ticket(
        description: str,
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> str:
        """创建人工工单,把问题转交人工客服处理。
        description: 问题描述(用户诉求的完整转述);
        ticket_type: 工单类型,售后/投诉/咨询 三选一"""
        conversation_id = current_conversation_id.get()
        if conversation_id is None:
            return "无法创建工单:缺少会话上下文。"
        async with session_factory() as session:
            ticket = await ticket_repo.create(
                session, conversation_id, description, ticket_type
            )
        return (
            f"已创建工单 {ticket.ticket_no}(类型:{ticket_type}),"
            f"人工客服会尽快跟进。问题描述:{description}"
        )

    return [query_faq, create_ticket]


@lru_cache
def build_default_registry() -> ToolRegistry:
    """全局工具清单:三 mock + 两 DB(session_factory 取 app.db 单例)。"""
    from app.db import get_sessionmaker

    registry = ToolRegistry()
    tools = [query_order, query_product, query_logistics]
    tools += build_db_tools(get_sessionmaker())
    for t in tools:
        registry.register(t)
    return registry
