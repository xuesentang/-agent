"""置信度闸(第五章第 7 点):知识类检索之后、进 Agent 之前。

为什么必须卡在这个位置:Agent 的答复是流式吐给用户的,等话说完再判「这条证据
够不够」已经晚了——用户已经读到。所以闸放在「检索完、还没开口」的空档里。

两层判定,顺序不能反:
1. **分数粗筛**(便宜):top1 相似度低于阈值直接判弱,连自评的一次模型调用都省掉;
2. **证据自评**(贵但准):分数过了才值得花一次调用问「这几条够不够答这个问题」。

阈值为什么单靠不住:实测 357 块库上,dense 相似度挤在 0.38~0.78 这个窄带里,
库外问题「W40 楦宽多少毫米」拿 0.62,比库内的「S23 支持快充吗」(0.49)还高。
所以阈值只用来挡明显离题的,真正区分「沾边」和「答得上」必须靠语义判定。
0.45 是这么定出来的:挡掉 0.386/0.446 两条纯离题,不让它继续往下走一步。

业务数据类不走这道闸:它没有检索证据,证据强弱无从谈起——拿订单工具查到的
订单号去判「证据够不够」是错的问法。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.citations import Citation, format_evidence
from app.rag.selfcheck import check_evidence

logger = logging.getLogger(__name__)

# 入池原因分级(与 low_confidence_questions.reason 一起存,复盘时分得清)
REASON_NO_HITS = "no_hits"
REASON_SCORE_LOW = "score_low"
REASON_SELF_CHECK = "self_check"
REASON_DEGRADED = "self_check_degraded"


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    detail: str = ""
    top_score: float = 0.0
    degraded: bool = False

    def pool_source(self) -> str:
        """落到 low_confidence_questions 的 source 字段(复用 CH04 的两个入口)。"""
        if self.reason in (REASON_NO_HITS, REASON_SCORE_LOW):
            return "retrieval_low_conf"
        return "self_check"

    def pool_reason(self) -> str:
        if self.detail:
            return f"{self.reason}: {self.detail}"[:255]
        return self.reason


def citations_from_payload(payloads: list[dict]) -> list[Citation]:
    """State 里只放 JSON 安全的 dict(checkpointer 要能序列化),取用时就地还原。"""
    return [Citation(**p) for p in payloads]


async def check_gate(
    question: str,
    citations: list[Citation],
    *,
    min_score: float,
    model=None,
) -> GateResult:
    """证据够不够把问题交给 Agent。判弱即拒答,后面的节点不再执行。"""
    if not citations:
        return GateResult(False, REASON_NO_HITS, "检索零命中")

    top = citations[0].score
    if top < min_score:
        return GateResult(
            False,
            REASON_SCORE_LOW,
            f"top1={top:.4f} < 阈值 {min_score:.2f}",
            top_score=top,
        )

    verdict = await check_evidence(question, format_evidence(citations), model=model, on_error="block")
    if verdict.useful:
        return GateResult(True, "", verdict.reason, top_score=top, degraded=verdict.degraded)

    reason = REASON_DEGRADED if verdict.degraded else REASON_SELF_CHECK
    return GateResult(False, reason, verdict.reason, top_score=top, degraded=verdict.degraded)
