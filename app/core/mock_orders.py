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
from datetime import date, timedelta

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
