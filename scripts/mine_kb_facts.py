"""为「干扰文档」批量挖事实(E2 补充)。

为什么只挖短的:
    让模型直接写 1000 字正文,单批要 60-90 秒且会被上游挂住(实测并发 4/6 直接零产出);
    但让它一次写 8 条一句话事实,两三秒就回来。所以把生成拆成「短事实 + 确定性渲染」:
    事实走模型(快),篇幅走模板(免费),两者都不依赖长输出。

产出:`data/kb_facts/<file>.json`,内容是可提交的小文件(便于复现,不必重跑模型)。

约束:干扰文档**不得提到核心型号(W/S/T/LP/TN/SB)与核心政策数值**(7 天、15 天、99 元、
129 元、8 元、12 元、保修 12 个月),否则会和主线政策打架,污染评估。

用法:
    python scripts/mine_kb_facts.py                 # 生成全部干扰文档的事实
    python scripts/mine_kb_facts.py --force         # 重新生成
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm import get_bulk_model

OUT_DIR = Path("data/kb_facts")
FACTS_PER_SECTION = 9
CONCURRENCY = 3

# 干扰品类:与核心三个品类(鞋靴/小家电/户外)相邻但不重叠。
DISTRACTOR_DOCS: list[dict] = [
    {
        "file": "maternal_faq.md",
        "title": "母婴用品常见问题",
        "content_type": "faq",
        "category": "母婴",
        "sections": ["奶瓶与喂养", "纸尿裤与湿巾", "婴儿洗护", "辅食与餐具", "婴儿衣物"],
    },
    {
        "file": "stroller_manual.md",
        "title": "婴儿推车与安全座椅手册",
        "content_type": "manual",
        "category": "母婴",
        "sections": ["推车型号与承重", "折叠与收纳尺寸", "安全座椅安装方式", "适用年龄与体重", "清洁与保养"],
    },
    {
        "file": "digital_faq.md",
        "title": "数码配件常见问题",
        "content_type": "faq",
        "category": "数码配件",
        "sections": ["充电与发热", "数据线兼容性", "无线耳机连接", "移动电源容量", "手机壳与贴膜"],
    },
    {
        "file": "charger_manual.md",
        "title": "充电器与数据线规格手册",
        "content_type": "manual",
        "category": "数码配件",
        "sections": ["功率与协议", "接口类型", "线材规格", "多口分配策略", "安全认证"],
    },
    {
        "file": "cleaning_faq.md",
        "title": "家居清洁用品常见问题",
        "content_type": "faq",
        "category": "家居",
        "sections": ["洗衣液与柔顺剂", "厨房去油", "地板清洁", "除菌与消毒", "衣物去渍"],
    },
    {
        "file": "storage_manual.md",
        "title": "收纳与床品规格手册",
        "content_type": "manual",
        "category": "家居",
        "sections": ["收纳箱尺寸", "床品面料与支数", "被子克重与适用温度", "衣柜挂杆承重", "清洁与防潮"],
    },
    {
        "file": "food_faq.md",
        "title": "食品与保健常见问题",
        "content_type": "faq",
        "category": "食品",
        "sections": ["保质期与储存", "配料与过敏原", "冲泡与食用方法", "临期商品说明", "食品类售后限制"],
    },
    {
        "file": "cookware_manual.md",
        "title": "厨房锅具手册",
        "content_type": "manual",
        "category": "家居",
        "sections": ["材质与适用炉具", "锅具尺寸对照", "不粘涂层使用", "开锅与养锅", "清洁禁忌"],
    },
    {
        "file": "bedding_manual.md",
        "title": "家纺尺寸与材质手册",
        "content_type": "manual",
        "category": "家居",
        "sections": ["四件套尺寸对照", "枕芯高度选择", "面料材质说明", "缩水与色差", "洗涤方式"],
    },
    {
        "file": "giftcard_policy.md",
        "title": "礼品卡与定制服务说明",
        "content_type": "policy",
        "category": "增值服务",
        "sections": ["礼品卡使用范围", "礼品卡有效期与退款", "刻字定制流程", "定制商品售后", "企业采购流程"],
    },
]

BANNED = (
    "W20", "W40", "W60", "S10", "S20", "T5", "LP100", "LP200", "LP300", "LP500",
    "TN2", "TN4", "SB200", "SB400", "129 元", "99 元包邮", "15 天无理由", "7 天无理由",
)

PROMPT = """你在为电商品牌「岚野 LANYE」写知识库的**非核心品类**文档,品类是「{category}」,
文档《{title}》里有一节叫「{section}」。

请写 {n} 条关于这一节的事实,要求:
- 每条一句话,20 至 45 字,必须含具体数值(尺寸/容量/时长/比例/温度等);
- 像真实电商文档里的规定,不要写成广告;允许啰嗦但每条只讲一件事;
- 不要提任何商品型号代码(W/S/T/LP/TN/SB 开头的编号一律不许出现);
- 不要提退换货期限、包邮门槛、运费金额、保修时长——这些属于核心政策,不在这里写;
- 不要输出编号、不要解释,一行一条。

只输出 JSON:{{"facts": ["...", "..."]}}"""


def _model():
    return get_bulk_model(max_tokens=1024, timeout=120)


def _parse(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    facts = payload.get("facts") or []
    return [str(f).strip() for f in facts if str(f).strip()]


async def _mine_section(model, doc: dict, section: str, sem: asyncio.Semaphore) -> list[str]:
    async with sem:
        for _ in range(2):
            resp = await model.ainvoke(
                [
                    (
                        "human",
                        PROMPT.format(
                            category=doc["category"],
                            title=doc["title"],
                            section=section,
                            n=FACTS_PER_SECTION,
                        ),
                    )
                ],
                response_format={"type": "json_object"},
            )
            facts = _parse(getattr(resp, "content", "") or "")
            clean = [f for f in facts if not any(b in f for b in BANNED)]
            if len(clean) >= FACTS_PER_SECTION // 2:
                return clean
        return clean


async def main_async(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _model()
    sem = asyncio.Semaphore(CONCURRENCY)
    for doc in DISTRACTOR_DOCS:
        path = OUT_DIR / f"{doc['file']}.json"
        if path.exists() and not args.force:
            print(f"skip {path}")
            continue
        sections = await asyncio.gather(
            *(_mine_section(model, doc, s, sem) for s in doc["sections"])
        )
        payload = {
            "file": doc["file"],
            "title": doc["title"],
            "content_type": doc["content_type"],
            "category": doc["category"],
            "sections": [
                {"h2": s, "facts": facts} for s, facts in zip(doc["sections"], sections, strict=True)
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        total = sum(len(s["facts"]) for s in payload["sections"])
        print(f"written {path}({total} 条事实)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="挖干扰文档的事实(短输出)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
