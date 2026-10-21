"""建库双写流水线(ch03 spec D5,提示词第三章「双写落库」)。

状态机:文档 → 切分 → MySQL(pending)→ embed → Milvus upsert → 回填 vector_id(done)。
- 按主键幂等:Milvus 主键 = chunk 自增 id,upsert 同键覆盖不重复;
- 挂了能重跑:任何一步中断,重跑只捞 vectorize_status=pending 的块补向量化;
- 整库重跑不重复插入:按 (category, questions, answer) 内容键去重。

中断演练:stop_after_batches=N 跑完 N 批后抛 BuildInterrupted(模拟 Ctrl-C),
重跑(不带该参数)自动补齐——验收标准 2 的单测版在这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.models.knowledge_chunk import KnowledgeChunk
from app.rag.chunker import RawChunk, split_document
from app.repositories import knowledge_repo


class BuildInterrupted(RuntimeError):
    """模拟中断:已处理的批次保留,剩余 pending 留给重跑。"""


@dataclass
class BuildReport:
    docs: int = 0
    chunks_inserted: int = 0
    chunks_skipped_dupe: int = 0
    chunks_vectorized: int = 0
    batches: int = 0
    interrupted: bool = False

    def summary(self) -> str:
        return (
            f"docs={self.docs} inserted={self.chunks_inserted} "
            f"skipped_dupe={self.chunks_skipped_dupe} vectorized={self.chunks_vectorized} "
            f"batches={self.batches} interrupted={self.interrupted}"
        )


# ---------------------------------------------------------------- 三格映射


def _split_qa(body: str) -> tuple[str, str]:
    """FAQ 正文拆问法/答案:Q:/问: 行是问法,A:/答: 行是答案,各自剥前缀;其余行进答案。"""
    questions: list[str] = []
    answers: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Q:", "问:")):
            questions.append(stripped.split(":", 1)[1].strip())
        elif stripped.startswith(("A:", "答:")):
            answers.append(stripped.split(":", 1)[1].strip())
        elif stripped:
            answers.append(stripped)
    return "\n".join(questions), "\n".join(answers)


def _parent_path(section_path: str) -> str:
    """上级标题路径:去掉最后一级;单级路径时返回自身。"""
    parts = [p for p in section_path.split(" > ") if p]
    return " > ".join(parts[:-1]) if len(parts) > 1 else (parts[0] if parts else "")


def to_knowledge_chunks(
    raw_chunks: list[RawChunk], content_type: str, doc_category: str = ""
) -> list[KnowledgeChunk]:
    """RawChunk → KnowledgeChunk 三格映射(提示词第三章落库结构)。

    - faq:questions = 真实问法(Q: 行),category = 文档分类或章节路径;
    - policy/manual:questions = 所在章节标题(末级),category = 上级标题路径;
    - 元数据四类(section_path/content_type/is_key_clause/前后指针)只存不进向量。
    """
    out: list[KnowledgeChunk] = []
    for rc in raw_chunks:
        if content_type == "faq":
            questions, answer = _split_qa(rc.body)
            if not questions:
                questions = rc.heading or rc.section_path
                answer = rc.body
            category = doc_category or rc.section_path
        else:
            questions = rc.heading or rc.section_path
            answer = rc.body
            category = _parent_path(rc.section_path)
        out.append(
            KnowledgeChunk(
                category=category,
                questions=questions,
                answer=answer,
                section_path=rc.section_path,
                content_type=content_type,
                is_key_clause=rc.is_key_clause,
            )
        )
    return out


@dataclass
class VectorizeReport:
    vectorized: int = 0
    batches: int = 0
    interrupted: bool = False


async def vectorize_pending(
    session,
    store,
    embedder,
    *,
    batch_size: int = 32,
    stop_after_batches: int | None = None,
    report: VectorizeReport | None = None,
) -> VectorizeReport:
    """双写第二步:捞 pending → embed → upsert → 回填 done。

    从 `build_index` 里抽出来的原因:ch09 审核通过也要写知识库,而「写知识库」
    的唯一正确姿势就是这条双写流程(建 pending → 向量化 → 回填)。复制一份
    逻辑到飞轮里迟早会走偏,所以抽成函数,两边调同一个。

    `report` 传进来就地累加:中断抛异常时调用方仍能看到已经处理了多少。
    """
    report = report or VectorizeReport()
    while True:
        pending = await knowledge_repo.get_pending(session, limit=batch_size)
        if not pending:
            break
        vectors = embedder.embed_texts([c.embedding_text() for c in pending])
        store.upsert(
            [
                {
                    "chunk_id": c.id,
                    "embedding": v,
                    # text 供 BM25 索引(关键词召回那一路),category 供元数据过滤;
                    # 两者都随块一起写,避免二次回表。
                    "text": c.embedding_text(),
                    "category": c.category or "",
                }
                for c, v in zip(pending, vectors)
            ]
        )
        for chunk in pending:
            await knowledge_repo.mark_done(session, chunk.id, str(chunk.id))
        report.vectorized += len(pending)
        report.batches += 1
        if stop_after_batches is not None and report.batches >= stop_after_batches:
            report.interrupted = True
            raise BuildInterrupted(f"中断于第 {report.batches} 批(剩 pending 待重跑补齐)")
    return report


# ---------------------------------------------------------------- 建库主流程


async def build_index(
    *,
    session_factory,
    store,
    embedder,
    docs: list[tuple[Path, str, str]],  # (path, content_type, doc_category)
    batch_size: int = 32,
    stop_after_batches: int | None = None,
) -> BuildReport:
    """离线建库入口。docs 为空时只补跑 pending(纯向量化恢复模式)。

    store/embedder 可注入(FakeMilvus/FakeEmbedder 单测用),不传则用真实现。
    """
    from app.rag.embeddings import LocalBGEEmbedder
    from app.rag.vectorstore import MilvusStore

    store = store or MilvusStore()
    embedder = embedder or LocalBGEEmbedder()
    store.ensure_collection()

    report = BuildReport()

    # 1. 文档 → 切分 → 三格映射 → 内容去重 → MySQL pending
    async with session_factory() as session:
        existing = await knowledge_repo.existing_chunk_keys(session)
        new_chunks: list[KnowledgeChunk] = []
        # 按文档分组:前后块指针只在同文档内链,跨文档首尾都断开
        doc_groups: list[list[KnowledgeChunk]] = []
        for path, content_type, doc_category in docs:
            report.docs += 1
            raw = split_document(Path(path).read_text(encoding="utf-8"))
            group: list[KnowledgeChunk] = []
            for chunk in to_knowledge_chunks(raw, content_type, doc_category):
                key = (chunk.category, chunk.questions, chunk.answer)
                if key in existing:
                    report.chunks_skipped_dupe += 1
                    continue
                existing.add(key)
                group.append(chunk)
                new_chunks.append(chunk)
            if group:
                doc_groups.append(group)
        if new_chunks:
            # 插入 + 链指针同一事务:不会留下半条链
            await knowledge_repo.add_chunks(session, new_chunks, commit=False)
            await knowledge_repo.link_neighbors(session, doc_groups, commit=False)
            await session.commit()
            report.chunks_inserted = len(new_chunks)

    # 2. 捞 pending → embed → upsert → 回填 done(中断后重跑从此继续)
    vr = VectorizeReport()
    async with session_factory() as session:
        try:
            await vectorize_pending(
                session,
                store,
                embedder,
                batch_size=batch_size,
                stop_after_batches=stop_after_batches,
                report=vr,
            )
        finally:
            # 中断也是「跑到哪了」的一部分,计数先搬进 report 再让异常往上走
            report.chunks_vectorized = vr.vectorized
            report.batches = vr.batches
            report.interrupted = vr.interrupted
    return report
