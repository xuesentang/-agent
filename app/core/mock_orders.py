"""订单 mock 数据的唯一产地(CH06 抽出)。

为什么单独一个模块:订单选择器要把「候选订单」渲染成卡片,卡片上的商品、金额、状态
必须和 `query_order` 查出来的一字不差——两处各自生成,用户点了卡片再去查单,看到
的却是另一件商品,这种不一致比没有选择器更糟。

所以数据生成只在这里做一次,工具和选择器都从 `order_view` 取同一份。
确定性依旧:同一个 order_id 永远同一结果(seed 固定),跨天、跨进程都一样。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

ORDER_PRODUCTS = [
    ("纯棉宽松圆领T恤", 89.00),
    ("无线蓝牙降噪耳机", 399.00),
    ("轻弹缓震跑步鞋", 329.00),
    ("大容量双肩通勤包", 159.00),
    ("智能运动手表", 699.00),
    ("山羊绒混纺围巾", 259.00),
]
ORDER_STATUS = ["待发货", "已发货", "运输中", "已签收", "已完成"]

# 固定时间基准:由 seed 派生偏移、不用 today(),保证同一入参跨天调用逐字一致
MOCK_EPOCH_DATE = date(2026, 1, 1)
MOCK_EPOCH_DT = datetime(2026, 6, 1, 8, 0, 0)

# 物流轨迹的阶段与城市(MCP 物流 Server 与内置工具共用这一份,避免两处不一致)
LOGISTICS_STAGES = [
    "已揽收",
    "运输中",
    "到达分拨中心",
    "离开分拨中心",
    "到达目的城市网点",
    "派送中",
    "已签收",
]
LOGISTICS_CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]

# 演示账号的订单号段。真实系统这里会是一次按 user_id 的查询;
# 本项目的订单是 mock 的,所以「该用户的订单」用 session 派生的固定号段模拟。
ORDER_ID_POOL = [str(i) for i in range(1001, 1021)]


@dataclass(frozen=True)
class OrderView:
    order_id: str
    product: str
    price: float
    quantity: int
    status: str
    order_date: str

    @property
    def amount(self) -> float:
        return round(self.price * self.quantity, 2)

    def to_payload(self) -> dict:
        """给前端的订单卡片字段。"""
        return {
            "order_id": self.order_id,
            "product": self.product,
            "quantity": self.quantity,
            "amount": self.amount,
            "status": self.status,
            "order_date": self.order_date,
        }


def order_view(order_id: str) -> OrderView:
    rng = random.Random(f"order:{order_id}")
    name, price = rng.choice(ORDER_PRODUCTS)
    qty = rng.randint(1, 3)
    status = rng.choice(ORDER_STATUS)
    order_date = (MOCK_EPOCH_DATE + timedelta(days=rng.randint(0, 300))).isoformat()
    return OrderView(
        order_id=str(order_id),
        product=name,
        price=price,
        quantity=qty,
        status=status,
        order_date=order_date,
    )


def candidate_orders(user_key: str, limit: int = 3) -> list[OrderView]:
    """「这个用户的最近订单」:用 user_key 派生固定几条,点开就是 query_order 的结果。"""
    rng = random.Random(f"user:{user_key}")
    picked = rng.sample(ORDER_ID_POOL, k=min(limit, len(ORDER_ID_POOL)))
    return [order_view(oid) for oid in picked]


def logistics_points(order_id: str) -> list[tuple[datetime, str, str]]:
    """物流轨迹点(时间从早到晚)。同入参结果确定,与订单号无关的随机源。

    MCP 物流 Server 与内置 `query_logistics` 都从这里取,保证「换了个实现,答出来的
    轨迹还是同一份」——否则第 8 章把物流交给 MCP 之后,同一个订单号会给出两套轨迹。
    """
    rng = random.Random(f"logistics:{order_id}")
    n = rng.randint(3, 6)
    start = rng.randint(0, len(LOGISTICS_STAGES) - n)
    stages = LOGISTICS_STAGES[start : start + n]
    points: list[tuple[datetime, str, str]] = []
    t = MOCK_EPOCH_DT + timedelta(hours=rng.randint(2, 240))
    for stage in reversed(stages):
        points.append((t, rng.choice(LOGISTICS_CITIES), stage))
        t -= timedelta(hours=rng.randint(4, 30))
    return points


def logistics_track_text(order_id: str) -> str:
    """物流轨迹的可读文本(最新在上)。"""
    lines = [f"订单 {order_id} 的物流轨迹(最新在上):"]
    lines += [f"- {t:%m-%d %H:%M} {city} {stage}" for t, city, stage in logistics_points(order_id)]
    return "\n".join(lines)


def warranty_info(order_id: str) -> tuple[bool, str]:
    """是否在保 + 到期日(售后 MCP Server 用)。按订单日期推 365 天。"""
    view = order_view(order_id)
    bought = date.fromisoformat(view.order_date)
    expires = bought + timedelta(days=365)
    return expires >= MOCK_EPOCH_DATE + timedelta(days=330), expires.isoformat()


def return_progress_text(order_id: str) -> str:
    """退货进度(售后 MCP Server 用)。按订单号派生一个固定阶段。"""
    rng = random.Random(f"return:{order_id}")
    stage = rng.choice(["待取件", "已取件", "运输中", "已入仓质检", "已退款", "已完成"])
    days = rng.randint(0, 6)
    return f"订单 {order_id} 的退货进度:{stage}(更新时间 {days} 天前)。"


def invoice_status_text(order_id: str) -> str:
    """发票状态(售后 MCP Server 用)。按订单号派生,固定不随机。"""
    view = order_view(order_id)
    rng = random.Random(f"invoice:{order_id}")
    kind = rng.choice(["电子普通发票", "增值税专用发票"])
    state = rng.choice(["已开具", "开具中", "未申请"])
    if state == "已开具":
        return f"订单 {order_id} 的{kind}已开具,金额 ¥{view.amount:.2f},可在订单详情页下载。"
    if state == "开具中":
        return f"订单 {order_id} 的{kind}正在开具,一般 1~3 个工作日完成,金额 ¥{view.amount:.2f}。"
    return f"订单 {order_id} 还没申请发票,可在订单详情页提交开票申请(金额 ¥{view.amount:.2f})。"
