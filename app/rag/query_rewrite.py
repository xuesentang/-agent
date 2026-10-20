"""Query 理解(ch04):口语问法归一 + 同义词扩展。

两条腿,职责分开:
- 改写归一(canonical):LLM 把「这东西多久能到我手上啊」这种口语压成一句标准问法,
  dense 向量用这句去检索——语义检索不怕长句,但归一能消掉口语噪声;
- 同义词扩展(keywords):只在**检索侧**把「邮费」扩成「邮费 运费」喂给 BM25,
  知识原文一个字都不改。入库侧拆存多份会让同一块知识出现多条向量、去重失效、
  引用编号指回同一段原文却编成多个号。
改写失败(模型报错/返回坏 JSON)一律回落原句,检索不能因为理解层挂了而不可用。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 电商客服高频同义词/近义词表:key 命中 query 就把 value 里的词追加进 BM25 查询。
# 表是白的:宁可少扩,不引入噪声(扩错一个词会把无关块顶上来)。
SYNONYMS: dict[str, tuple[str, ...]] = {
    "邮费": ("运费", "快递费", "配送费"),
    "运费": ("邮费", "快递费", "配送费"),
    "快递": ("物流", "配送", "发货"),
    "物流": ("快递", "配送", "运单"),
    "质保": ("保修", "三包", "保修期"),
    "保修": ("质保", "三包", "维修"),
    "退货": ("退款", "退掉", "七天无理由"),
    "退款": ("退货", "退钱", "原路返回"),
    "换货": ("更换", "换一个", "调换"),
    "发票": ("开票", "抬头", "税号"),
    "发货": ("出库", "寄出", "什么时候发"),
    "尺码": ("码数", "偏大", "偏小", "选码"),
    "库存": ("有没有货", "现货", "缺货"),
    "优惠": ("折扣", "满减", "券"),
    "维修": ("修理", "返修", "保修"),
    "到账": ("退到", "到账时间", "多久到"),
}


@dataclass
class RewrittenQuery:
    """理解层的产物:原句 / 归一问法 / BM25 扩展串 / 扩了哪些词(留痕用)。"""

    raw: str
    canonical: str
    keywords: list[str] = field(default_factory=list)

    @property
    def bm25_query(self) -> str:
        """BM25 用串 = 归一问法 + 扩展词。去重保序,不重复加噪声。"""
        parts: list[str] = [self.canonical]
        for kw in self.keywords:
            if kw and kw not in self.canonical:
                parts.append(kw)
        return " ".join(parts)


def expand_synonyms(query: str) -> list[str]:
    """同义词扩展(检索侧):命中表就展开,没命中返回空列表。"""
    out: list[str] = []
    for key, values in SYNONYMS.items():
        if key in query:
            for v in values:
                if v not in out:
                    out.append(v)
    return out


REWRITE_PROMPT = """把用户的口语问法改写成一句标准、完整、可检索的客服问句。

要求:
- 只保留原意,不要补充用户没说的信息,不要回答问题;
- 去掉语气词、寒暄、情绪表达;
- 保留关键实体:商品型号、订单号、金额、时效等原样保留;
- 输出 JSON:{{"canonical": "标准问法", "keywords": ["同义或近义检索词", ...]}}
  keywords 只放有助于检索的补充词,没有就给空数组。

用户问法:{question}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


async def rewrite_query(question: str, model=None) -> RewrittenQuery:
    """LLM 归一 + 静态同义词表扩展。model 为 None 时只做同义词扩展(不调模型)。

    LLM 输出坏 JSON / 调用失败 → canonical 回落原句,keywords 只留同义词表的部分。
    """
    keywords = expand_synonyms(question)
    if model is None:
        return RewrittenQuery(raw=question, canonical=question, keywords=keywords)

    from app.config import get_settings

    settings = get_settings()
    method = settings.llm_structured_method
    try:
        if method == "json_mode":
            resp = await model.ainvoke(
                [("human", REWRITE_PROMPT.format(question=question))],
                response_format={"type": "json_object"},
            )
            payload = _parse_json(getattr(resp, "content", "") or "")
            canonical = (payload.get("canonical") or "").strip()
            extra = payload.get("keywords") or []
        else:  # function_calling / 默认:直接让模型按 schema 出结构化结果
            from pydantic import BaseModel

            class _Rewrite(BaseModel):
                canonical: str
                keywords: list[str] = []

            structured = model.with_structured_output(_Rewrite, method=method)
            result = await structured.ainvoke(
                [("human", REWRITE_PROMPT.format(question=question))]
            )
            canonical = (result.canonical or "").strip()
            extra = result.keywords
    except Exception as exc:  # noqa: BLE001 - 理解层失败不阻断检索
        logger.warning("问法改写失败,回落原句:%s", exc)
        return RewrittenQuery(raw=question, canonical=question, keywords=keywords)

    if not canonical:
        canonical = question
    merged = list(keywords)
    for kw in extra:
        if isinstance(kw, str) and kw.strip() and kw.strip() not in merged:
            merged.append(kw.strip())
    return RewrittenQuery(raw=question, canonical=canonical, keywords=merged)


def _parse_json(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
