"""knowledge_chunks 表数据访问(ch03 RAG 原文权威源)。

双写状态机的 MySQL 侧操作:建库写 pending、捞 pending 补向量化、回填 done、
在线检索按 id 取原文;挖知识去重时按规范化问法查既有 questions。
"""

from __future__ import annotations

import re

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_chunk import KnowledgeChunk


async def add_chunks(
    session: AsyncSession, chunks: list[KnowledgeChunk], *, commit: bool = True
) -> list[KnowledgeChunk]:
    """批量落库(默认 pending),返回带自增 id 的实例。

    commit=False 时只 flush 不提交,由调用方把多步写操作收进同一事务
    (建库的插入 + 链指针、挖知识的暂存 + 知识入库)。
    """
    session.add_all(chunks)
    if commit:
        await session.commit()
    else:
        await session.flush()
    for chunk in chunks:
        await session.refresh(chunk)
    return chunks


async def get_pending(session: AsyncSession, limit: int = 100) -> list[KnowledgeChunk]:
    """捞待向量化的块(id 升序,重跑可补;limit 控制单批规模)。"""
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.vectorize_status == "pending")
        .order_by(KnowledgeChunk.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def mark_done(session: AsyncSession, chunk_id: int, vector_id: str) -> None:
    """回填 Milvus 主键 + 状态转 done(双写最后一步;失败留给下次重跑补)。"""
    chunk = await session.get(KnowledgeChunk, chunk_id)
    if chunk is None:
        return
    chunk.vector_id = vector_id
    chunk.vectorize_status = "done"
    await session.commit()


async def reset_vector_status(session: AsyncSession) -> int:
    """把全部块退回 pending(配合「删掉向量集合重建」使用)。

    为什么必须有这一步:向量集合被 drop 之后,MySQL 里的 vector_id 还写着旧值、
    状态还是 done,于是重跑时这些块永远不会被重新向量化——库里明明有 357 行,
    Milvus 只有 336 条,检索时那 21 块等于不存在。这个缺陷只有在「重建集合」
    这条路径上才会暴露,平时的增量补跑完全正常。
    """
    result = await session.execute(
        update(KnowledgeChunk)
        .where(KnowledgeChunk.vectorize_status != "pending")
        .values(vectorize_status="pending", vector_id=None)
    )
    await session.commit()
    return int(result.rowcount or 0)


async def get_by_ids(session: AsyncSession, ids: list[int]) -> list[KnowledgeChunk]:
    """按 id 取原文(在线检索命中后回 MySQL 拿三格文本)。

    返回顺序与入参 ids 无关(按 id 升序),调用方需要自行按检索分序重排。
    """
    if not ids:
        return []
    stmt = select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)).order_by(KnowledgeChunk.id)
    return list((await session.execute(stmt)).scalars().all())


async def link_neighbors(
    session: AsyncSession, groups: list[list[KnowledgeChunk]], *, commit: bool = True
) -> None:
    """按文档分组回填前后块指针(同文档内相邻;跨文档边界不链)。

    groups 每个元素是一篇文档切出的块列表,组内顺序即文档顺序,
    组内 prev/next 两两相连,每组首块的 prev 与末块的 next 都是 None。
    分组必须由调用方给:chunk 表没有文档路径字段,光看列表顺序分不出文档边界。
    """
    for group in groups:
        for i, chunk in enumerate(group):
            chunk.prev_chunk_id = group[i - 1].id if i > 0 else None
            chunk.next_chunk_id = group[i + 1].id if i < len(group) - 1 else None
    if commit:
        await session.commit()


def normalize_question(text: str) -> str:
    """问法规范化:去空白/标点/大小写,用于挖知识去重的精确比对(ch04 才上语义去重)。"""
    return re.sub(r"[\s\W_]+", "", text).lower()


async def existing_questions(session: AsyncSession) -> set[str]:
    """知识库现有全部问法(规范化后),挖知识入库前比对用。

    演示量级(几百块)可全量拉;数据量上来后应改按 keyword 预筛(ch04+ 优化点)。
    """
    stmt = select(KnowledgeChunk.questions)
    rows = (await session.execute(stmt)).scalars().all()
    out: set[str] = set()
    for raw in rows:
        for q in raw.splitlines():
            nq = normalize_question(q)
            if nq:
                out.add(nq)
    return out


async def existing_chunk_keys(session: AsyncSession) -> set[tuple[str, str, str]]:
    """现有 (category, questions, answer) 内容键集合——建库整库重跑的去重依据。

    DDL 没有文档路径字段,按内容键去重是幂等重建的最简口径;
    同内容不同章节路径会被判重跳过(可接受:内容相同则检索结果相同)。
    """
    stmt = select(KnowledgeChunk.category, KnowledgeChunk.questions, KnowledgeChunk.answer)
    return set((await session.execute(stmt)).all())
