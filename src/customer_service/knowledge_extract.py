"""Offline staging and deduplication for conversation-derived QA."""

import re
import json
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import select

from customer_service.knowledge_store import KnowledgeStore, NewChunk
from customer_service.models import Conversation, QAExtractionStaging
from customer_service.repository import Repository
from pydantic import BaseModel


class ExtractedPair(BaseModel):
    source_ref: str
    question: str
    answer: str


class ExtractedBatch(BaseModel):
    pairs: list[ExtractedPair]


class LLMQAExtractor:
    def __init__(self, model, method: str = "json_mode") -> None:
        self.model = model.with_structured_output(ExtractedBatch, method=method)

    def extract(self, conversations):
        payload = [{"source_ref": cid, "messages": [{"role": row.role, "content": row.content} for row in rows if row.role in {"user", "assistant"} and row.content]} for cid, rows in conversations]
        prompt = (
            "从以下客服对话抽取可复用的用户问题和有依据的客服答案，只输出 JSON，格式为 "
            "{\"pairs\":[{\"source_ref\":\"会话ID\",\"question\":\"...\",\"answer\":\"...\"}]}。"
            "不要凭空编造政策。没有可靠答案的对话不提取。每条 source_ref 必须来自输入。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = self.model.invoke(prompt)
        if not isinstance(result, ExtractedBatch):
            result = ExtractedBatch.model_validate(result)
        return [(item.source_ref, item.question, item.answer) for item in result.pairs]


def _norm(text: str) -> str:
    return re.sub(r"\s+|[？?。.!！]", "", text).strip().lower()


def stage_pairs(session_factory, batch_no: str, source_ref: str, pairs: list[tuple[str, str]]) -> None:
    with session_factory.begin() as session:
        session.add_all(QAExtractionStaging(batch_no=batch_no, source_ref=source_ref, question=q.strip(), answer=a.strip()) for q, a in pairs if q.strip() and a.strip())


def promote_staged(session_factory, store: KnowledgeStore) -> int:
    with session_factory() as session:
        rows = session.scalars(select(QAExtractionStaging).order_by(QAExtractionStaging.id)).all()
        seen: set[tuple[str, str]] = set()
        kept = []
        dispositions = {}
        for row in rows:
            key = (_norm(row.question), _norm(row.answer))
            if row.status == "kept":
                seen.add(key)
            elif row.status == "extracted" and key in seen:
                dispositions[row.id] = "discarded"
            elif row.status == "extracted":
                seen.add(key)
                dispositions[row.id] = "kept"
                kept.append((row.question, row.answer, row.source_ref or f"staging:{row.id}"))
    chunks = [NewChunk("历史对话", question, answer, f"qa:{sha256((_norm(question) + chr(0) + _norm(answer)).encode('utf-8')).hexdigest()}") for question, answer, _ in kept]
    store.add_chunks(chunks)
    with session_factory.begin() as session:
        for row_id, status in dispositions.items():
            session.get(QAExtractionStaging, row_id).status = status
    return len(chunks)


def extract_batches(session_factory, extractor, batch_size: int = 20) -> int:
    """Read conversation history in small batches and stage extracted pairs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    repo = Repository(session_factory)
    count = 0
    last_id = ""
    while True:
        with session_factory() as session:
            ids = session.scalars(select(Conversation.id).where(Conversation.id > last_id).order_by(Conversation.id).limit(batch_size)).all()
        if not ids:
            break
        last_id = ids[-1]
        batch = [(cid, repo.list_messages(cid)) for cid in ids]
        pairs = extractor.extract(batch)
        batch_no = uuid4().hex
        staged = []
        for source_ref, question, answer in pairs:
            if source_ref not in ids:
                raise ValueError("Extractor returned unknown source conversation")
            if question.strip() and answer.strip():
                staged.append(QAExtractionStaging(batch_no=batch_no, source_ref=source_ref, question=question.strip(), answer=answer.strip()))
        with session_factory.begin() as session:
            session.add_all(staged)
        count += len(staged)
    return count
