"""证据充分性自评(ch04 生成质量控制)。

位置很关键:**在生成之前**判。答复是流式吐给用户的,等答案出来再判「刚才那段
是不是编的」已经晚了——只能事后道歉,用户已经读到了。所以在检索完、进生成之前
先问模型一句:手里这几条证据,够不够回答这个问题?

判「不够」的三种典型:证据讲的是别的事、只沾了点边、粒度对不上(问具体型号,
证据只有品类政策)。判定失败(模型挂了/坏 JSON)一律当「够」——宁可放过,
不可误杀,否则一次上游抖动会让所有正常问题都被拒答。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SELF_CHECK_PROMPT = """你是客服知识库的质量审核。判断下面这些检索到的证据,是否足以支撑回答用户的问题。

判断标准(严格):
- 证据必须直接讲这个问题问的事,只沾边、只同属一个大类都不算够;
- 问具体型号/具体商品,证据只有通用政策的,算不够;
- 证据里有明确答案(规则、数字、流程、时效),才算够。

只输出 JSON:{{"useful": true 或 false, "reason": "一句话说明判断依据"}}
不要回答用户的问题,不要给建议。

用户问题:{question}

证据:
{evidence}"""


@dataclass
class SelfCheckResult:
    """自评结果。useful=False 时 reason 会写进低置信度问题池,留作复盘。"""

    useful: bool
    reason: str = ""
    degraded: bool = False  # 判定本身失败(不是证据不够),用于区分日志口径


async def check_evidence(question: str, evidence: str, model=None) -> SelfCheckResult:
    """证据够不够答这个问题。判不了就当够(degraded=True)。"""
    if not evidence.strip():
        return SelfCheckResult(useful=False, reason="没有任何检索证据")

    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("自评模型不可用,跳过证据自评:%s", exc)
            return SelfCheckResult(useful=True, reason="", degraded=True)

    from app.rag.query_rewrite import _parse_json  # 同一套宽松 JSON 解析

    try:
        resp = await model.ainvoke(
            [("human", SELF_CHECK_PROMPT.format(question=question, evidence=evidence))],
            response_format={"type": "json_object"},
        )
        payload = _parse_json(getattr(resp, "content", "") or "")
    except TypeError:
        # 端点不支持 response_format 参数时退回普通调用
        try:
            resp = await model.ainvoke(
                [("human", SELF_CHECK_PROMPT.format(question=question, evidence=evidence))]
            )
            payload = _parse_json(getattr(resp, "content", "") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("证据自评失败,按「够」处理:%s", exc)
            return SelfCheckResult(useful=True, reason="", degraded=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("证据自评失败,按「够」处理:%s", exc)
        return SelfCheckResult(useful=True, reason="", degraded=True)

    if "useful" not in payload:
        return SelfCheckResult(useful=True, reason="", degraded=True)
    useful = bool(payload.get("useful"))
    reason = str(payload.get("reason") or "").strip()
    return SelfCheckResult(useful=useful, reason=reason)


# 拒答话术:明确说没查到,不硬编;并给出可行动的下一步(说细一点 / 转人工)。
ABSTAIN_REPLY = (
    "抱歉,这个问题我在知识库里没有找到可靠的依据,不想凭猜测给您答案。"
    "您可以再补充一下具体型号或订单信息,我帮您核实;需要人工介入的话我也可以帮您建工单。"
)
