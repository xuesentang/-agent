# 客服聊天工具调用 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本项目无 Git 仓库，用户已要求不做 Git 操作；聊天页徽章按用户约定采用 Vibe Coding，不套 brainstorm、TDD、code review。

**Goal:** 让现有 DeepSeek 客服聊天在一轮内选择并执行至多一个工具，把调用及结果存入 MySQL，再流式给出最终答复。

**Architecture:** FastAPI 路由只负责 HTTP/SSE；`ChatService` 负责一次工具决策和最终生成；工具注册与执行器隔离业务工具；Repository 用 SQLAlchemy 2.x 持久化四表。同步数据库与工具调用在异步聊天链路中交给线程执行，避免阻塞 token 流。

**Tech Stack:** Python 3.12、FastAPI、LangChain `@tool`、langchain-openai、SQLAlchemy 2.x、PyMySQL、本机 MySQL80、pytest、httpx。

**Spec:** `docs/superpowers/specs/2026-10-06-customer-service-tools-design.md`

## Global Constraints

- 只使用现有 DeepSeek OpenAI 兼容接入；不换模型栈。本章不做 Agent Loop、RAG 或真实电商/物流 API，也不建订单、商品、物流表。
- 每轮最多执行一个工具调用，随后用未绑定工具的模型生成最终回答；最终文本仍按上游 chunk 发 SSE `token`。
- 工具为 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`；所有工具用 LangChain `@tool` 和 Pydantic 输入 Schema。
- 数据库只建 `faq`、`conversations`、`messages`、`tickets` 四表；会话 `user_id` 可选，默认 `guest`；原有 `conversation_id` 外部格式不变。
- 使用本机已安装的 MySQL80（`127.0.0.1:3306`）及独立的 `customer_service` 数据库；不依赖 Docker。用户自行在本机 `.env` 填写 `DATABASE_URL`，密码不进入计划、代码或日志。此条是 2026-10-06 用户确认的方案修订。
- 只读工具超时后最多重试一次；`create_ticket` 失败不自动重试。失败结果回灌模型并明确告知用户，不泄露堆栈。
- SQL `LIKE` 只查 `faq.question`；“邮费是多少”未命中为预期结果，必须在真实评估中留痕。
- 每项任务完成时立即追记 `dev-notes/ch02.md` 的用户关键原话、产出、纠偏、翻车与返工；计划评审、code review、finish 也分别即时追记。
- 无 Git 操作。动用新的库/框架/API 前先查 Context7；已查 SQLAlchemy 2.x ORM/事务、LangChain 工具、FastAPI 流式响应，实施细节再按需查。

## File map

| 文件 | 责任 |
| --- | --- |
| `.env.example`, `pyproject.toml` | 本机 MySQL 连接示例与依赖 |
| `src/customer_service/db.py`, `models.py`, `repository.py` | SQLAlchemy engine/session、四表映射、查询与持久化 |
| `src/customer_service/seed.py` | 显式建表与幂等 FAQ 样例灌入 |
| `src/customer_service/business_tools.py` | 五个 `@tool` 与 Pydantic Schema |
| `src/customer_service/tool_executor.py` | 注册表、校验、超时、重试与错误结果 |
| `src/customer_service/chat.py`, `history.py`, `prompts.py`, `main.py`, `config.py` | 一次工具决策、历史裁剪、SSE 与配置 |
| `src/customer_service/index.html` | 聊天气泡中的工具徽章；不增加工具面板 |
| `tests/test_db.py`, `test_business_tools.py`, `test_tool_executor.py`, `test_tool_chat.py` | SQLite/假模型离线验证；保留第一章测试并按新接口更新 |
| `evals/ch02_cases.jsonl`, `evals/run_ch02.py` | 三项真实 DeepSeek 验收及预期漏召回记录 |
| `README.md`, `dev-notes/ch02.md` | 启动、演示、结果与过程记录 |

## 跨任务接口约定

`Repository` 每个方法内部新建并关闭 `Session`；公开方法只返回已经脱离 Session 的 DTO 或标量，不能把 ORM 对象交给异步聊天链路。`append_messages` 接收 `MessageRow(role: Literal['user','assistant','tool'], content: str, tool_calls: list[dict] | None = None, tool_call_id: str | None = None)`，按传入顺序一次事务写入。`list_messages` 返回同形状 DTO，并按 `(created_at, id)` 排序。`search_faq` 返回 `FAQRow(question, answer, category)`；`create_ticket` 返回 `TicketRow(ticket_id, conversation_id, description, ticket_type, status)`。

`ChatService.stream` 只产出 `ChatEvent(name: Literal['tool_status','token'], data: dict)`；`session`、`done`、`error` 由 HTTP 路由包装。决策模型的原始 `AIMessage.tool_calls` 要写入 assistant 行；工具输出写入 `role='tool'` 行并保留相同 `tool_call_id`。历史重建把这一对转成 `AIMessage(tool_calls=...)` 和 `ToolMessage(tool_call_id=...)`。无工具的决策文本不作为回答或消息记录，最终答复统一由未绑定工具的模型流式生成。

测试用 SQLite 临时文件、固定随机种子和假模型；真实 MySQL/DeepSeek 只用于在线验收。API 与浏览器对外仍用既有 `conversation_id` 字段，新增 `user_id` 默认为 `guest`。

## Task 1：数据库骨架与显式初始化

**Files:** Create `src/customer_service/db.py`, `models.py`, `repository.py`, `seed.py`, `tests/test_db.py`; Modify `pyproject.toml`, `.env.example`, `src/customer_service/config.py`。

**Interfaces:** `make_engine(database_url: str) -> Engine`；`SessionFactory = sessionmaker`；`Base.metadata` 含四表；`Repository(session_factory)` 提供 `create_conversation(user_id: str = 'guest') -> str`、`get_conversation(conversation_id: str) -> ConversationRow | None`、`list_messages(conversation_id: str) -> list[MessageRow]`、`append_messages(conversation_id: str, rows: list[MessageRow]) -> None`、`search_faq(keyword: str) -> list[FAQRow]`、`create_ticket(conversation_id: str, description: str, ticket_type: str) -> TicketRow`；`init_database(engine) -> None` 建表并幂等灌 FAQ。

- [ ] **Step 1：先写失败测试。** 用 SQLite 临时文件构造 engine，断言 `Base.metadata.tables` 恰为四表；`init_database` 两次后 FAQ 不重复；创建会话默认 `guest`；插入 assistant 的 `tool_calls` JSON 和 tool 的 `tool_call_id` 后按 id 顺序读回；工单关联会话且工单号唯一；`search_faq('退货政策')` 命中，`search_faq('邮费')` 为空。示例断言：

```python
assert set(Base.metadata.tables) == {"faq", "conversations", "messages", "tickets"}
assert [m.role for m in repo.list_messages(cid)] == ["user", "assistant", "tool"]
assert repo.search_faq("邮费") == []
```

- [ ] **Step 2：运行 `.venv\Scripts\python.exe -m pytest tests/test_db.py -q -p no:cacheprovider`，确认因新接口缺失失败。**
- [ ] **Step 3：实现四表与 Repository。** 用 `DeclarativeBase`、`Mapped`、`mapped_column`，`messages.role` 用数据库约束限定 `user/assistant/tool`；外键关联并给 `messages(conversation_id,id)` 建索引。所有写操作用 `SessionFactory.begin()` 原子提交；`search_faq` 用 `FAQ.question.like(f"%{escaped_keyword}%", escape="\\")`，先转义 `%`、`_`、`\\`，避免用户输入被当通配符。初始化只由 `seed.py` 命令触发。
- [ ] **Step 4：加入本机 MySQL 配置与依赖。** `.env.example` 给出 `mysql+pymysql://...@127.0.0.1:3306/customer_service?charset=utf8mb4` 示例；`pyproject.toml` 增 `SQLAlchemy>=2,<3`、`PyMySQL>=1,<2`。只在独立项目数据库中建表，不修改已有其他库。
- [ ] **Step 5：运行 Task 1 测试，待用户在 `.env` 配好本机数据库账号后，在本机 MySQL80 上显式初始化并核对四表与 FAQ 数量；立即追记任务结果。**

## Task 2：五个业务工具

**Files:** Create `src/customer_service/business_tools.py`, `tests/test_business_tools.py`; Create `evals/ch02_cases.jsonl`。

**Interfaces:** `build_business_tools(repository: Repository, conversation_id: str, rng: Random) -> list[BaseTool]` 返回五个已绑定当前会话的工具。输入 Schema：`OrderInput(order_id: str)`、`ProductInput(product_name: str)`、`FAQInput(keyword: str)`、`TicketInput(description: str, ticket_type: str)`。输出统一为可 JSON 序列化的文本，模拟工具含 `source: "demo"`。

- [ ] **Step 1：先写标注样例与失败测试。** `evals/ch02_cases.jsonl` 至少有物流 1001、退货政策、邮费三个输入及预期工具；用固定 `Random(7)` 断言三个模拟工具返回请求标识、`source=demo` 和必要字段；FAQ 两个查询按样例命中/未命中；`create_ticket` 写入一个关联当前会话的工单。纯 Prompt/标注内容先用样例评估，不对文案写镜像单测。
- [ ] **Step 2：运行 `.venv\Scripts\python.exe -m pytest tests/test_business_tools.py -q -p no:cacheprovider`，确认新接口缺失失败。**
- [ ] **Step 3：实现 `@tool(args_schema=...)`。** 模拟数据只在工具内部生成，不建表或调用外部 API；FAQ 调 `repository.search_faq`；工单调 `repository.create_ticket`。工具说明要求 `query_faq.keyword` 传核心名词短语，如“退货政策”，避免把“是什么”一并传给 SQL `LIKE`；工具内部不做同义词扩展或语义匹配。所有字段说明写在 Schema 和工具 docstring 中；`query_faq` 的“未找到”使用明确标记，如 `{ "found": false, "matches": [] }`。
- [ ] **Step 4：运行 Task 2 测试；核对五个工具名称、Schema 及标注样例；立即追记任务结果。**

## Task 3：工具注册、校验与执行策略

**Files:** Create `src/customer_service/tool_executor.py`, `tests/test_tool_executor.py`。

**Interfaces:** `ToolExecutor(tools: list[BaseTool], timeout_seconds: float = 5).execute(call: dict) -> ToolMessage`（异步方法），`ToolMessage.tool_call_id` 必须与调用 id 一致；`registry` 为按名称映射的只读表。

- [ ] **Step 1：先写失败测试。** 未知工具返回可回灌错误；缺参/错参被 Pydantic 拦截且业务函数未运行；只读工具首轮超时、第二轮成功时恰执行两次；`create_ticket` 抛错时只执行一次；每个结果都保留调用 id。示例：

```python
result = await executor.execute({"name": "query_faq", "args": {"keyword": "退货政策"}, "id": "call_1"})
assert result.tool_call_id == "call_1"
```

- [ ] **Step 2：运行 Task 3 测试，确认接口缺失失败。**
- [ ] **Step 3：实现最小执行器。** 从注册表选工具；先 `tool.args_schema.model_validate(call['args'])`；用 `asyncio.to_thread` 执行同步工具，并以 `asyncio.wait_for` 设超时；只读工具最多再试一次，`create_ticket` 不重试；将未知名称、`ValidationError`、`TimeoutError`、其他执行异常转为 `ToolMessage(status='error')` 的安全文本。注意线程超时不能取消已开始的同步调用，因此 `create_ticket` 绝不在超时后重放；将此限制写 README。
- [ ] **Step 4：运行 Task 3 测试；立即追记任务结果。**

## Task 4：工具链接入现有 SSE 聊天

**Files:** Modify `src/customer_service/chat.py`, `history.py`, `prompts.py`, `main.py`, `config.py`; Create `tests/test_tool_chat.py`; Update `tests/test_chat_stream.py`, `tests/test_history.py`。

**Interfaces:** `ChatService(model, repository: Repository, tool_factory, token_budget: int)`；`prepare(conversation_id: str, message: str) -> list[BaseMessage]`；`stream(conversation_id: str, message: str) -> AsyncIterator[ChatEvent]`；路由保留 `POST /chat/stream`，新增可选 `user_id`，发 `session`、`tool_status`、`token`、`done`/`error`。

- [ ] **Step 1：先写失败测试。** 假模型决策返回 `AIMessage(tool_calls=[...])`，断言事件顺序 `session, tool_status(running), tool_status(success), token..., done`；无工具时无 `tool_status`；含工具轮次写库顺序 user/assistant/tool/assistant 且 `tool_call_id` 配对；最终回答中断时无最终 assistant；模型给出多个工具调用时不执行任何一个并发 `error`；旧会话仍可加载上下文，未知 id 为 404。用 SQLite 临时库与 fake model，不调用真实 API。
- [ ] **Step 2：运行 Task 4 测试，确认旧 `ChatService` 不符合事件/持久化接口而失败。**
- [ ] **Step 3：实现单轮状态机。** 先持久化 user，再 `bind_tools(tools, tool_choice='auto', parallel_tool_calls=False)` 做一次 `ainvoke` 决策；若有一个调用，先发 running，调用 `ToolExecutor.execute` 并保存 assistant/tool；追加配对 `ToolMessage`，再用未绑定工具的 `model.astream` 流式最终答复；无调用时直接走未绑定模型 `astream`。成功完成后写最终 assistant。数据库操作通过 `asyncio.to_thread` 调 Repository，每次操作创建自己的 Session，不跨线程共享 Session。若决策或执行后流中断，已写的 user 与完整调用/结果保留，历史重建仍保持消息配对。
- [ ] **Step 4：重做历史裁剪。** 按完整轮次裁剪，工具调用申请与 tool 结果作为不可拆开的同一组；系统 Prompt、当前消息和 512 输出 token 预留仍纳入预算。Prompt 加规则：仅根据工具结果描述状态，模拟查询信息要注明演示数据；FAQ 未命中时说明未查到，不编造政策；转人工时才可建工单。
- [ ] **Step 5：运行 Task 4 与既有测试；若 DeepSeek 不接受 `parallel_tool_calls=False` 或当前 `bind_tools` 参数，先查 Context7 和上游错误，再请用户评审兼容性方案，不自行换技术栈；立即追记任务结果。**

## Task 5：聊天页工具徽章（Vibe Coding 例外）

**Files:** Modify `src/customer_service/index.html`。

- [ ] **Step 1：直接在 SSE `handleEvent` 处理中接收 `tool_status`。** 当 `state=running` 时在本轮助手气泡的元信息旁创建徽章并显示“查询中”；`success/error` 更新同一个徽章。不展示工具参数或原始 JSON。无工具轮次不出现徽章。
- [ ] **Step 2：实际浏览器试物流、FAQ 与无工具对话，检查徽章只属于当前气泡、手机不溢出、SSE 回复仍逐块显示。** 本步骤按用户指定不套 brainstorm/TDD/code review，发现效果问题直接迭代。完成后立即追记任务结果。

## Task 6：真实 MySQL + DeepSeek 验收和交付

**Files:** Create `evals/run_ch02.py`; Modify `README.md`, `dev-notes/ch02.md`。

- [ ] **Step 1：README 给出本机 MySQL80 独立数据库创建、`.env` 配置、显式建表/灌数、Uvicorn 启动、三条浏览器提问和数据库核对命令。** 说明模拟数据、线程超时限制和 FAQ 预期漏召回。
- [ ] **Step 2：真实 DeepSeek 评估。** `evals/run_ch02.py` 通过 HTTP/SSE 对“订单 1001 的物流到哪了”“退货政策是什么”“邮费是多少”逐一检查工具事件与最终答复；同时核对第二句传入 `query_faq` 的关键词可命中 FAQ 表。第三句直接断言 `repository.search_faq('邮费') == []`，并把模型是否选择 `query_faq` 与无匹配分开报告。验收后查询 MySQL，核对四表、消息流水及 `tool_call_id`；单独演示一次明确转人工，确认一条工单。
- [ ] **Step 3：完整运行 `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`，做页面浏览器验收。按 `superpowers:requesting-code-review` 审查后修复 Critical/Important 并重测；立即追记 code review 结论。**
- [ ] **Step 4：按 `superpowers:verification-before-completion` 用最新测试输出核对结论，追记 finish 阶段，交付演示命令、测试结果、`dev-notes/ch02.md` 路径。**

## 计划自查

- 四表与种子数据：Task 1；五工具及 LIKE 漏召回：Task 2；Schema/超时/重试：Task 3；一轮 Function Calling、SSE、持久化：Task 4；页面徽章：Task 5；三项真实验收与工单：Task 6。
- 无新增订单/商品/物流表，不接真实业务 API，不做 Agent Loop/RAG；仅在本机 MySQL80 的独立项目数据库中建表。
- 外部依赖：用户在本机配置 MySQL `DATABASE_URL`、DeepSeek 实际工具调用能力。任何未通过的在线验收必须如实记录，不能以离线测试代替。
