"""意图识别:一个 prompt 判七类,再加一张分流表。

两件事分开写是有意的:
- **判什么**(意图)交给模型,它是模糊语义的活;
- **走去哪**(分流)写死在 `INTENT_ROUTES` 里,它是业务规矩的活。

混在一起的后果是:改一次分流要动 prompt,改一次 prompt 要重测分流,
两个都不稳定的东西互相拽着。表驱动之后,CH06 加「其他」兜底类只加一行。

解析失败不猜:落到「售后」(业务数据类,直进主力 Agent)。理由是这条路的失败
代价最小——闲聊类是零模型调用的固定话术,把没判出来的问题当寒暄等于直接放弃;
知识类要先过检索闸,判错就多一次拒答。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.query_rewrite import _parse_json

logger = logging.getLogger(__name__)

# 七类意图(提示词第 6 点),顺序即 prompt 里的选项顺序
INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")

# 四个出口
ROUTE_KNOWLEDGE = "knowledge"  # 先检索 → 过闸 → Agent
ROUTE_BUSINESS = "business"  # 不预检索,直接进 Agent
ROUTE_COMPLAINT = "complaint"  # 不进 Agent,安抚 + 给选项
ROUTE_CHITCHAT = "chitchat"  # 固定话术,零模型调用

INTENT_ROUTES: dict[str, str] = {
    "物流": ROUTE_BUSINESS,
    "订单": ROUTE_BUSINESS,
    "商品咨询": ROUTE_KNOWLEDGE,
    "退款退货": ROUTE_KNOWLEDGE,
    "售后": ROUTE_BUSINESS,
    "投诉": ROUTE_COMPLAINT,
    "闲聊": ROUTE_CHITCHAT,
}

FALLBACK_INTENT = "售后"

INTENT_PROMPT = """你是电商客服系统的意图分类器。把用户这句话归到下面七类之一。

类别(只能选这七个):
- 物流:查快递进度、问什么时候到、催发货、改地址
- 订单:查订单内容、金额、状态、改单取消
- 商品咨询:问商品参数、规格、型号、材质、价格、库存
- 退款退货:问退款/退货的政策、流程、时效、运费谁出
- 售后:已购商品出问题要维修、换货、补偿,或需要查售后进度
- 投诉:表达不满、要投诉、要求追责或赔偿
- 闲聊:打招呼、道谢、与购物无关的聊天

判断时以「用户想干什么」为准,不要被提到的商品或订单带偏:
- 「我买的鞋开胶了能退吗」→ 退款退货(问政策),不是售后
- 「我这个订单申请退货了,进度到哪了」→ 售后(查进度),不是退款退货
- 「你们这破快递我要投诉」→ 投诉
- 「订单1001的物流到哪了」→ 物流

只输出 JSON,不要解释:
{{"intent": "七类之一", "confidence": 0.0 到 1.0 之间的小数}}

用户这句话:{question}"""


@dataclass
class IntentResult:
    intent: str
    route: str
    confidence: float = 0.0
    fallback: bool = False  # 解析失败/模型异常时为 True,用于日志区分口径
    raw: str = ""


def route_of(intent: str) -> str:
    """未知意图一律按业务数据类处理(同上:失败代价最小的方向)。"""
    return INTENT_ROUTES.get(intent, ROUTE_BUSINESS)


async def classify_intent(question: str, *, model=None, history: list | None = None) -> IntentResult:
    """判七类。模型异常或 JSON 不可用 → 落 FALLBACK_INTENT,不抛异常打断图。"""
    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001 - 拿不到模型也要让图继续走
            logger.warning("意图识别模型不可用,归入 %s:%s", FALLBACK_INTENT, exc)
            return IntentResult(FALLBACK_INTENT, route_of(FALLBACK_INTENT), fallback=True)

    prompt = INTENT_PROMPT.format(question=question)
    if history:
        # 本章只把最近几轮当背景贴上去,真正的指代补全在 CH06
        recent = "\n".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in history[-4:])
        prompt = f"最近对话:\n{recent}\n\n{prompt}"

    try:
        resp = await model.ainvoke([("human", prompt)], response_format={"type": "json_object"})
        text = getattr(resp, "content", "") or ""
    except TypeError:
        # 端点不支持 response_format 时退回普通调用
        try:
            resp = await model.ainvoke([("human", prompt)])
            text = getattr(resp, "content", "") or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("意图识别失败,归入 %s:%s", FALLBACK_INTENT, exc)
            return IntentResult(FALLBACK_INTENT, route_of(FALLBACK_INTENT), fallback=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("意图识别失败,归入 %s:%s", FALLBACK_INTENT, exc)
        return IntentResult(FALLBACK_INTENT, route_of(FALLBACK_INTENT), fallback=True)

    return parse_intent(text)


def parse_intent(text: str) -> IntentResult:
    """把模型输出解析成结果。单独拆出来是为了能拿标注样例离线验一遍解析。"""
    payload = _parse_json(text)
    intent = str(payload.get("intent") or "").strip()
    if intent not in INTENT_ROUTES:
        # 模型可能给「退款」这种半截词,做一次前缀兜底匹配
        for known in INTENTS:
            if intent and (intent in known or known in intent):
                intent = known
                break
    if intent not in INTENT_ROUTES:
        logger.info("意图不在七类内,落兜底:%r", intent)
        return IntentResult(FALLBACK_INTENT, route_of(FALLBACK_INTENT), fallback=True, raw=text)

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return IntentResult(intent, route_of(intent), confidence=confidence, raw=text)
