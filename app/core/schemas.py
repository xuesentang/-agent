"""Structured output schema for after-sales ticket extraction (spec D5)."""

from typing import Literal

from pydantic import BaseModel, Field


class AfterSalesInfo(BaseModel):
    """售后描述抽取结果。三个字段均可空：用户未提及即 null，禁止编造。"""

    order_id: str | None = Field(
        default=None,
        description="订单号。用户未提及或无法确定完整订单号时为 null，禁止猜测或编造。",
    )
    claim_type: Literal[
        "refund",        # 退款
        "return_refund",  # 退货退款
        "exchange",       # 换货
        "repair",         # 维修
        "compensation",   # 补偿/赔偿
        "complaint",      # 投诉
        "inquiry",        # 咨询
        "other",          # 其他
    ] | None = Field(
        default=None,
        description="诉求类型。refund=退款，return_refund=退货退款，exchange=换货，repair=维修，"
        "compensation=补偿/赔偿，complaint=投诉，inquiry=咨询，other=其他。无法判断时为 null。",
    )
    expected_solution: str | None = Field(
        default=None,
        description="用户期望的解决方案。用户未明确表达期望时为 null。",
    )
