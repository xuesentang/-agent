"""对话挖知识(ch03 T5,提示词第三章「离线建库·对话挖知识」)。

流程:历史客服对话分批喂 LLM 抽 QA 对 → qa_extraction_staging(extracted)
→ 整体去重(规范化问法 vs staging 已有 + knowledge_chunks 现有 questions)
→ kept 入 knowledge_chunks(content_type=qa_mined,待 build_knowledge 补向量化)
/ discarded 标注。CLI:scripts/mine_qa.py;定时执行:FastAPI lifespan 开关。

设计要点:
- extractor 可注入(FakeExtractor 单测),默认 GLM(json_mode 结构化输出);
- 抽取输出脏数据(缺字段/空问答)跳过不崩;
- 去重口径与知识库一致(normalize_question 去空白标点),同义不同写法
  的语义去重留 ch04;
- dry_run 全流程只读不写,用于先看一批能抽出什么;
- 重复执行无害:再跑一遍抽出的还是那些对,全部判重 discarded。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_chunk import KnowledgeChunk
from app.models.message import Message
from app.models.qa_extraction_staging import QaExtractionStaging
from app.repositories import knowledge_repo, qa_staging_repo

# 挖出来的 QA 进 knowledge_chunks 用的固定分类标签:conversations 表没有
# 分类字段,用来源标签撑住 category 这一格(真正的类目归一并入 ch09 飞轮)。
MINED_CATEGORY = "对话挖掘"

# 单条对话原文进 prompt 的字符上限(超长截断,防一批把 context 撑爆)
_TRANSCRIPT_MAX_CHARS = 6000


class QaPair(BaseModel):
    """一条抽出的问答对。"""

    question: str = Field(description="用户的真实问法,可适度规范化口语")
    answer: str = Field(description="客服给出的有效答复")


class QaExtractionResult(BaseModel):
    """LLM 结构化输出外层(强制 JSON)。"""

    pairs: list[QaPair] = Field(default_factory=list, description="抽取到的问答对,没有则空列表")


@dataclass
class MineReport:
    """一轮挖知识的结果账本。"""

    batches: int = 0
    conversations: int = 0
    extracted: int = 0
    kept: int = 0
    discarded: int = 0

    def summary(self) -> str:
        return (
            f"batches={self.batches} conversations={self.conversations} "
            f"extracted={self.extracted} kept={self.kept} discarded={self.discarded}"
        )


# ---------------------------------------------------------------- 抽取器


QA_EXTRACTION_PROMPT = """你是一个客服知识库挖掘助手。下面是一段用户与客服的历史对话,请从中提取有价值、可复用的问答对。

要求:
- 只提取事实性知识:政策规则、操作步骤、费用说明、售后流程等;寒暄、闲聊、情绪表达、个人化信息不要;
- question 用用户的真实问法(可适度规范化口语,保留原意);
- answer 用客服答复中的有效内容,去掉冗余客套;
- 一段对话没有可提炼的知识时,pairs 返回空列表;
- 严格按 JSON 输出。

对话内容:
{transcript}"""


def _format_transcript(messages: list[Message]) -> str:
    """消息流水 → 纯文本对话实录(user/assistant/tool 三角色)。"""
    role_names = {"user": "用户", "assistant": "客服", "tool": "工具结果"}
    lines: list[str] = []
    for msg in messages:
        name = role_names.get(msg.role, msg.role)
        content = (msg.content or "").strip()
        if not content and msg.tool_calls:
            content = f"[调用工具] {msg.tool_calls}"
        if content:
            lines.append(f"{name}: {content[:_TRANSCRIPT_MAX_CHARS]}")
    return "\n".join(lines)


async def default_llm_extractor(messages: list[Message]) -> list[QaPair]:
    """默认抽取器:GLM(json_mode)把对话实录压成 QA 对。"""
    transcript = _format_transcript(messages)
    if not transcript.strip():
        return []
    from app.config import get_settings
    from app.core.llm import get_extract_model

    model = get_extract_model()
    structured = model.with_structured_output(
        QaExtractionResult, method=get_settings().llm_structured_method
    )
    result = await structured.ainvoke(
        [("human", QA_EXTRACTION_PROMPT.format(transcript=transcript))]
    )
    if isinstance(result, dict):
        result = QaExtractionResult.model_validate(result)
    return result.pairs


def _clean_pairs(pairs: list[QaPair]) -> list[QaPair]:
    """过滤脏数据:空问答、问法答法相同的丢掉。"""
    out: list[QaPair] = []
    for pair in pairs:
        q, a = pair.question.strip(), pair.answer.strip()
        if not q or not a or q == a:
            continue
        out.append(QaPair(question=q, answer=a))
    return out


# ---------------------------------------------------------------- 数据源


async def list_minable_conversation_ids(session: AsyncSession) -> list[int]:
    """有用户消息的会话 id 升序——挖知识的数据源。"""
    from app.models.conversation import Conversation

    stmt = (
        select(Conversation.id)
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == "user")
        .distinct()
        .order_by(Conversation.id)
    )
    return list((await session.execute(stmt)).scalars().all())


# ---------------------------------------------------------------- 主流程


async def mine_qa_pairs(
    *,
    session_factory,
    batch_size: int = 20,
    max_batches: int | None = None,
    extractor=None,
    dry_run: bool = False,
) -> MineReport:
    """一轮挖知识。extractor 默认 default_llm_extractor,可注入 FakeExtractor。

    分批:batch_size 个会话一批,batch_no 用时间戳(按批追溯、防串味);
    去重:规范化问法 vs staging 全部存量 + knowledge_chunks 现有 questions,
       seen 集合跨批累积(本批接受的问法,同批后续/下一批再出现即判重);
    入库:kept 写 knowledge_chunks(vectorize_status=pending),
      接下来跑 build_knowledge --pending-only 即可补向量化。
    """
    extractor = extractor or default_llm_extractor
    report = MineReport()
    batch_no = datetime.now().strftime("%Y%m%d%H%M%S%f")

    async with session_factory() as session:
        conv_ids = await list_minable_conversation_ids(session)
        # seen 初始 = 库里已有的全部问法(staging 三个状态 + 知识库),跨批累积
        seen = set(await knowledge_repo.existing_questions(session))
        for status in ("extracted", "kept", "discarded"):
            rows = await qa_staging_repo.list_by_status(session, status)
            seen.update(knowledge_repo.normalize_question(r.question) for r in rows)

    for start in range(0, len(conv_ids), batch_size):
        if max_batches is not None and report.batches >= max_batches:
            break
        batch_ids = conv_ids[start : start + batch_size]
        report.batches += 1
        report.conversations += len(batch_ids)
        source_ref = f"conversations:{batch_ids[0]}-{batch_ids[-1]}"

        async with session_factory() as session:
            # 1. 逐会话抽取(单会话抽坏跳过,不炸整轮)
            pairs: list[tuple[str, str]] = []
            for conv_id in batch_ids:
                messages = (
                    (
                        await session.execute(
                            select(Message)
                            .where(Message.conversation_id == conv_id)
                            .order_by(Message.id)
                        )
                    )
                    .scalars()
                    .all()
                )
                try:
                    extracted = await extractor(messages)
                except Exception:  # noqa: BLE001 - 单会话失败跳过,不炸整轮
                    continue
                # 抽取器输出的统一净化口:脏数据(空问答/问答相同)在这里拦掉,
                # 不管 extractor 是默认 LLM 还是注入的测试替身。
                pairs.extend((p.question, p.answer) for p in _clean_pairs(extracted))
            report.extracted += len(pairs)

            # 2. 分类(先判重再落库——判重看见的是插入前的存量)
            classified: list[tuple[str, str, bool]] = []  # (question, answer, is_kept)
            for question, answer in pairs:
                key = knowledge_repo.normalize_question(question)
                is_kept = bool(key) and key not in seen
                if is_kept:
                    seen.add(key)
                classified.append((question, answer, is_kept))
            kept = [item for item in classified if item[2]]
            report.kept += len(kept)
            report.discarded += len(classified) - len(kept)
            if not classified:
                continue

            # 3. 全量落 staging(extracted),再按分类翻状态
            if dry_run:
                continue
            # 3~4 步同一事务:暂存、翻状态、知识入库要么全成要么全不成。
            # 否则崩在中间会留下"暂存有 kept、知识库没这块"的洞,而重跑时
            # 该问法已进 seen,永远补不回来。
            rows = await qa_staging_repo.add_batch(
                session, batch_no, pairs, source_ref=source_ref, commit=False
            )
            kept_ids = [row.id for row, (_, _, is_kept) in zip(rows, classified) if is_kept]
            discarded_ids = [row.id for row, (_, _, is_kept) in zip(rows, classified) if not is_kept]
            await qa_staging_repo.set_status_bulk(session, kept_ids, "kept", commit=False)
            await qa_staging_repo.set_status_bulk(
                session, discarded_ids, "discarded", commit=False
            )

            # 4. kept 入知识库(pending,等 build_knowledge 补向量化)
            if kept_ids:
                await knowledge_repo.add_chunks(
                    session,
                    [
                        KnowledgeChunk(
                            category=MINED_CATEGORY,
                            questions=q,
                            answer=a,
                            content_type="qa_mined",
                        )
                        for q, a, _ in kept
                    ],
                    commit=False,
                )
            await session.commit()
    return report
