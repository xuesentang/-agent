"""指代消解 + Query 改写(第六章第 1、2 点)。

一次模型调用干两件事,因为它们读的是同一份上下文:
- **指代消解**——「它能退吗」补成「订单 1001 的无线蓝牙降噪耳机能退吗」,
  让这句话离开对话也能看懂;
- **Query 改写**——「邮费多少」归一成「运费标准」,给检索用。

分开调两次的代价不只是多一次请求:两次对「用户在说什么」的理解可能不一致,
于是检索用一个意思、回答用另一个意思。一次调用出一个结果,两处共用。

问题本身已经完整的**原样透传**(`changed=false`):改写不是目的,
不该为了「看起来在做事」把一句好问题揉一遍——揉完的检索效果往往更差。

同一份输出里还带一个槽位 `order_id`:它是「退款子流程要不要弹订单选择器」的判据。
让模型顺便抽,比事后正则捞一句自然语言可靠;正则只作模型失败时的兜底。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.rag.query_rewrite import _parse_json

logger = logging.getLogger(__name__)

_ORDER_ID_RE = re.compile(r"(?<!\d)(\d{4,20})(?!\d)")

UNDERSTAND_PROMPT = """你是电商客服系统的问句理解模块。结合最近对话,把用户这句话处理成两项结果。

一、指代消解:把「它 / 这个 / 那个 / 这单 / 上面那个 / 那双」这类指代,按最近对话补成完整的问题。
   - 补全后要能脱离对话单独看懂;
   - 补的是**对话里真实出现过**的东西,不许猜、不许编造商品或订单号;
   - 问题本来就完整、没有指代,就原样输出,并把 changed 置为 false。
   - 「最开始 / 第一次 / 早先 / 一开始」这类**时序指代**指向的是【早前对话梗概】里
     按时间先后出现的第一笔订单;梗概里的订单按时间先后排列,取最前面那一单的订单号。

二、问法归一(给检索用):把口语说法换成知识库里的标准说法,意思不能变。
   - 例:邮费 → 运费;保修 → 质保;能退吗 → 退货条件;多久到账 → 退款时效;
   - 只归一说法,**不要扩写、不要换问题、不要回答**,不要加入用户没说的条件。

三、槽位:抽出用户**在这句话或最近对话里明确说过的订单号**(纯数字)。没有就填 null,禁止猜。
   时序指代(「最开始那个订单」)消解出订单号后,这一项也要填上。

只输出 JSON:
{{"resolved": "补全后的问题", "canonical": "归一后的问法", "order_id": "订单号或 null", "changed": true 或 false}}

早前对话梗概(更早轮次已压缩,其中事实可信;订单按时间先后列出):
{summary}

最近对话:
{history}

用户这句话:{question}"""

EXPAND_PROMPT = """你是电商客服的检索查询扩写器。把用户这一句问题泛化成 2 到 3 条侧重点不同的检索查询,
用来在知识库里多路召回。

要求:
- 保持原意,不要新增用户没提的条件,不要回答问题;
- 每条都要能独立拿去检索,不要写成「同上」「另外」这种;
- 第一条必须贴近原句;
- 只输出 JSON,字段只有 queries:{{"queries": ["...", "..."]}}

用户问题:{question}"""

MAX_EXPANDED = 3


@dataclass
class UnderstoodQuery:
    raw: str
    resolved: str
    canonical: str
    order_id: str | None = None
    changed: bool = False
    fallback: bool = False  # 模型没给出可用结果时为 True


def guess_order_id(text: str) -> str | None:
    """兜底:直接从文本里捞 4 位以上数字(模型失败时用)。"""
    m = _ORDER_ID_RE.search(text or "")
    return m.group(1) if m else None


def history_text(history: list | None, limit: int = 6) -> str:
    """把历史渲染成一行行「角色:内容」。

    历史在 State 里是 **LangChain 消息对象**(`trim_history` 的产物),
    在 eval / 单测里是 dict——两种都要认。只认一种就会在真链路上炸:
    `dict.get` 打在 `HumanMessage` 上是 AttributeError。
    """
    if not history:
        return "(无)"
    lines: list[str] = []
    for msg in history[-limit:]:
        if isinstance(msg, dict):
            role, content = msg.get("role", ""), msg.get("content", "")
        else:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = str(getattr(msg, "content", "") or "")
        role = {"human": "用户", "ai": "客服", "user": "用户", "assistant": "客服"}.get(role, role)
        if content.strip():
            lines.append(f"{role}:{content[:200]}")
    return "\n".join(lines) or "(无)"


async def understand_query(
    question: str, *, history: list | None = None, summary: str = "", model=None
) -> UnderstoodQuery:
    """指代消解 + 问法归一。任何失败都退回原句,理解层不许把主链路带崩。

    `summary` 是早前对话的梗概:长会话里早期内容已经被压掉,不给它,
    「最开始那个订单后来怎么说」这种时序指代就只能靠层 1 里零星的单号瞎猜——
    实测会指到最近提过的那一单上,答出一个事实正确但**答非所问**的订单。
    """
    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("理解层模型不可用,原样透传:%s", exc)
            return _passthrough(question, fallback=True)

    prompt = UNDERSTAND_PROMPT.format(
        summary=(summary or "(无)"), history=history_text(history), question=question
    )
    try:
        resp = await model.ainvoke([("human", prompt)], response_format={"type": "json_object"})
        text = getattr(resp, "content", "") or ""
    except TypeError:
        try:
            resp = await model.ainvoke([("human", prompt)])
            text = getattr(resp, "content", "") or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("指代消解失败,原样透传:%s", exc)
            return _passthrough(question, fallback=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("指代消解失败,原样透传:%s", exc)
        return _passthrough(question, fallback=True)

    payload = _parse_json(text)
    resolved = str(payload.get("resolved") or "").strip()
    canonical = str(payload.get("canonical") or "").strip()
    if not resolved:
        return _passthrough(question, fallback=True)

    order_id = payload.get("order_id")
    order_id = str(order_id).strip() if order_id not in (None, "", "null") else None
    if order_id and not order_id.isdigit():
        order_id = None
    if not order_id:
        # 模型没抽到就自己捞一次:resolved 是补全后的句子,历史里的订单号已补进去了
        order_id = guess_order_id(resolved) or guess_order_id(question)

    return UnderstoodQuery(
        raw=question,
        resolved=resolved,
        canonical=canonical or resolved,
        order_id=order_id,
        changed=bool(payload.get("changed", resolved != question)),
    )


def _passthrough(question: str, *, fallback: bool) -> UnderstoodQuery:
    return UnderstoodQuery(
        raw=question,
        resolved=question,
        canonical=question,
        order_id=guess_order_id(question),
        changed=False,
        fallback=fallback,
    )


async def expand_queries(question: str, *, model=None) -> list[str]:
    """一句问题 → 多条检索查询。失败就只返回原句,扩写是加分项,不是前置条件。"""
    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("扩写模型不可用,只用原句检索:%s", exc)
            return [question]

    try:
        resp = await model.ainvoke(
            [("human", EXPAND_PROMPT.format(question=question))],
            response_format={"type": "json_object"},
        )
        payload = _parse_json(getattr(resp, "content", "") or "")
    except TypeError:
        try:
            resp = await model.ainvoke([("human", EXPAND_PROMPT.format(question=question))])
            payload = _parse_json(getattr(resp, "content", "") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询扩写失败,只用原句:%s", exc)
            return [question]
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询扩写失败,只用原句:%s", exc)
        return [question]

    raw = payload.get("queries")
    if not isinstance(raw, list):
        return [question]

    out: list[str] = []
    for item in raw:
        q = str(item or "").strip()
        if q and q not in out:
            out.append(q)
        if len(out) >= MAX_EXPANDED:
            break
    if question not in out:
        out.insert(0, question)
    return out[:MAX_EXPANDED]
