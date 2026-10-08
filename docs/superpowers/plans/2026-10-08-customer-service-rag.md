# Customer Service RAG Implementation Plan

> **For agentic workers:** Execute each task with a failing test first, verify, then record its outcome in `dev-notes/ch03.md`.

**Goal:** Replace FAQ literal lookup with BGE-M3 dense retrieval over Milvus while preserving the chat tool contract and providing recoverable offline ingestion.

**Architecture:** MySQL `knowledge_chunks` remains authoritative. An offline CLI writes pending chunks, upserts vectors by the same integer ID to Milvus, then marks rows done. Online retrieval searches Milvus IDs and reads answer text from MySQL. A separate offline extraction command stages conversation QA before global deduplication.

**Tech Stack:** Python 3.12, SQLAlchemy 2, LangChain, FlagEmbedding BGE-M3, PyMilvus, MySQL80, Zilliz Cloud Free.

**Spec:** `docs/superpowers/specs/2026-10-08-customer-service-rag-design.md`

## Global constraints

- Existing `query_faq(keyword)` input and `{found, matches}` output stay compatible.
- Only dense BGE-M3 vectors; no keyword recall, hybrid search or reranking.
- Keep secrets in local `.env`; never commit credentials or model weights.
- Use the existing Windows Python environment; do not migrate the project to WSL or Docker.
- Record each completed task immediately in `dev-notes/ch03.md`.

## Task 1: Knowledge schema and ingestion identity

**Files:** `src/customer_service/models.py`, `src/customer_service/knowledge_store.py`, `tests/test_knowledge_store.py`.

- [ ] Write failing tests for two new tables, source/content deduplication, pending-to-done transitions and idempotent import.
- [ ] Add `KnowledgeChunk` and `QAExtractionStaging` ORM models with the agreed reference fields plus `source_key` and `content_hash` unique identity.
- [ ] Add repository operations that insert pending chunks, preserve IDs on re-import, list pending rows and mark a row done only after vector upsert.
- [ ] Verify task tests and append `dev-notes/ch03.md`.

## Task 2: Markdown and FAQ normalization

**Files:** `src/customer_service/knowledge_parser.py`, `tests/test_knowledge_parser.py`, `knowledge/*.md`.

- [ ] Write failing tests for heading paths, recursive long section splitting, sentence-boundary overlap, and table-row split with repeated headers.
- [ ] Implement pure parser returning category/questions/answer/metadata; add a small realistic shipping-policy sample for the stated acceptance case.
- [ ] Verify tests and append the task record.

## Task 3: Dense vector store and resumable indexing

**Files:** `src/customer_service/vector_store.py`, `src/customer_service/knowledge_ingest.py`, `tests/test_knowledge_ingest.py`, `pyproject.toml`, `.env.example`.

- [ ] Write failing tests using fake embedder and fake Milvus client. Simulate an interruption after upsert and verify retry updates the same primary key and marks it done.
- [ ] Implement BGE-M3 adapter with lazy loading, 1024-dimensional dense vectors and PyMilvus adapter using explicit integer IDs and COSINE metric.
- [ ] Implement CLI to import Markdown/FAQ, index pending rows and report counts; configure only via local env vars.
- [ ] Verify tests and append the task record.

## Task 4: Online query_faq integration

**Files:** `src/customer_service/business_tools.py`, `src/customer_service/chat.py`, `src/customer_service/main.py`, `src/customer_service/knowledge_search.py`, `tests/test_rag_tool.py`.

- [ ] Write failing tests for semantic hit mapping, no hit, unavailable vector service as explicit tool error, and unchanged `{found, matches}` output.
- [ ] Inject retriever into the existing tool builder and chat service; fetch only `done` rows by Milvus IDs.
- [ ] Update tool description for semantic search and verify no other tool changes.
- [ ] Verify tests and append the task record.

## Task 5: Conversation knowledge extraction

**Files:** `src/customer_service/knowledge_extract.py`, `tests/test_knowledge_extract.py`, README.

- [ ] Write failing tests for batched conversation input, staging first, global normalized deduplication and repeat-safe promotion.
- [ ] Implement offline CLI using the configured LLM; provide Windows Task Scheduler command without automatically changing the user's scheduler.
- [ ] Verify tests and append the task record.

## Task 6: Real integration and review

**Files:** README, `evals/run_ch03.py`, `dev-notes/ch03.md`.

- [ ] Inspect the target MySQL schema before creating only the two new tables.
- [ ] With user-provided Zilliz endpoint and token in `.env`, create the `knowledge` collection, ingest examples, deliberately interrupt after vector upsert, rerun and verify one vector per row.
- [ ] Run real BGE-M3 retrieval for “邮费是多少” and a browser chat turn; run all offline tests.
- [ ] Review the diff against the spec, fix actionable issues and record finish evidence.
