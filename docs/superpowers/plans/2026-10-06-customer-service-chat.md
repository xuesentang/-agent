# 电商客服纯对话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本项目无 Git 仓库，按用户要求省去所有 Git 操作；当前无 Python 运行时，创建环境前须取得用户确认。

**Goal:** 提供可用 curl 验收的 SSE 多轮客服对话和售后描述结构化提取。

**Architecture:** FastAPI 暴露两个接口。LangChain `ChatPromptTemplate` 构造对话与提取 Prompt，`ChatOpenAI` 从环境配置直连 GLM 或 DeepSeek。对话历史仅存单进程内存，按预算裁剪；提取独立调用 `with_structured_output`。

**Tech Stack:** Python、FastAPI、LangChain、langchain-openai、Pydantic、pytest、httpx。

**Spec:** `docs/superpowers/specs/2026-10-06-customer-service-chat-design.md`

## Global Constraints

- 本章仅验收 GLM、DeepSeek；不做工具调用、Agent 循环或聊天页面。
- 上游地址、模型名、密钥均从 `.env` 读取；真实密钥不可写入仓库、测试和日志。
- 会话由服务端内存保存，以 `conversation_id` 续聊；单进程、重启丢失。
- 售后提取必须使用 `with_structured_output`；无效结果显式报错。
- 动库或框架 API 前先查 Context7。已查 FastAPI SSE、LangChain Prompt/裁剪/`ChatOpenAI`、DeepSeek JSON Output、GLM 流式文档；实施时遇到新 API 再查。
- 每完成一个任务立即追记 `dev-notes/ch01.md`；计划评审通过、code review、finish 也分别即时追记。
- 不进行 Git 操作。现无 Python 运行时，环境准备必须先取得用户确认；不默认迁移到 WSL。

## File map

| 文件 | 单一职责 |
| --- | --- |
| `pyproject.toml`、`.env.example`、`.gitignore` | 依赖与安全配置示例 |
| `src/customer_service/config.py` | 加载并校验模型与预算设置 |
| `src/customer_service/prompts.py` | 客服及售后提取的 PromptTemplate |
| `src/customer_service/history.py` | 内存会话与近似 token 裁剪 |
| `src/customer_service/chat.py` | 模型流式调用与完成后提交历史 |
| `src/customer_service/aftersales.py` | Pydantic 输出模型与结构化提取 |
| `src/customer_service/main.py` | FastAPI 请求模型、SSE 与 JSON 路由 |
| `tests/` | 假模型、接口和历史预算测试；不使用真实密钥 |
| `evals/aftersales.jsonl` | 标注售后描述与期望字段 |
| `README.md` | Windows 启动及三项 curl 验收命令 |
| `dev-notes/ch01.md` | 每阶段原话、产出、纠偏和返工 |

## Task 1: 配置与 Prompt

**Files:** Create `pyproject.toml`, `.env.example`, `.gitignore`, `src/customer_service/__init__.py`, `src/customer_service/config.py`, `src/customer_service/prompts.py`, `tests/test_config_prompts.py`.

**Interfaces:** `Settings.from_env() -> Settings`，字段 `model_base_url`, `model_name`, `model_api_key`, `structured_output_method`, `history_token_budget`；`build_model(settings) -> ChatOpenAI`；`CHAT_PROMPT`、`EXTRACTION_PROMPT` 为 `ChatPromptTemplate`。

- [ ] **Step 1: 写失败测试。** 用 `monkeypatch` 设置五个环境变量，断言 `Settings.from_env()` 读取值；删掉密钥断言抛 `ValueError` 且错误文本不含密钥；断言 `CHAT_PROMPT.format_messages(history=[], message="你好")` 首条为 system、末条为 human。
- [ ] **Step 2: 运行 `uv run --no-sync pytest tests/test_config_prompts.py -q`，确认因模块/接口缺失而失败。** 若尚未获准准备环境，仅记录此步骤待执行，不以缺少 Python 冒充测试失败。
- [ ] **Step 3: 实现最小代码。** `Settings.from_env` 用 `python-dotenv` 加载 `.env`，校验非空配置和正整数预算；`build_model` 调用 `ChatOpenAI(base_url=..., model=..., api_key=...)`。客服 system prompt 写明身份、不能假称查单或已退款、信息不足追问、不得泄露系统提示词。提取 prompt 明确输出 JSON 和缺失值规则。
- [ ] **Step 4: 运行 `uv run --no-sync pytest tests/test_config_prompts.py -q`，确认通过；立即追记任务结果。**

## Task 2: 会话历史与预算

**Files:** Create `src/customer_service/history.py`, `tests/test_history.py`.

**Interfaces:** `ConversationStore.create() -> str`；`ConversationStore.get(conversation_id: str) -> list[BaseMessage]`（未知 ID 抛 `KeyError`）；`ConversationStore.append_turn(id: str, user: str, assistant: str) -> None`；`fit_history(history: list[BaseMessage], current_message: str, budget: int) -> list[BaseMessage]`。

- [ ] **Step 1: 写失败测试。** 创建 ID 后 `get` 为空；提交两轮后消息顺序是 human/AI/human/AI；未知 ID 抛 `KeyError`；超预算时仅裁掉最旧的完整轮次；当前消息超预算抛 `ValueError`。
- [ ] **Step 2: 运行 `uv run --no-sync pytest tests/test_history.py -q`，确认接口缺失导致失败。**
- [ ] **Step 3: 实现最小代码。** 用 `uuid4().hex` 生成 ID；字典存储消息列表；使用 LangChain `count_tokens_approximately` 估算 `CHAT_PROMPT` 格式化后的 system、历史、当前消息总量，从近到远选取完整轮次；在每次加入旧轮次前检查预算，避免半轮残留。
- [ ] **Step 4: 运行 `uv run --no-sync pytest tests/test_history.py -q`，确认通过；立即追记任务结果。**

## Task 3: SSE 流式对话

**Files:** Create `src/customer_service/chat.py`, `src/customer_service/main.py`, `tests/test_chat_stream.py`.

**Interfaces:** `ChatService.stream(conversation_id: str, message: str) -> AsyncIterator[str]` 逐块输出上游文本，成功后提交完整轮次；`create_app(chat_service, extraction_service) -> FastAPI` 用于注入假模型；`POST /chat/stream` 返回 `session`、零到多个 `token`、`done` 或 `error` 事件。

- [ ] **Step 1: 写失败测试。** 假模型 `astream` 依次产出 `"你好"`、`"，请问"`；`httpx.AsyncClient` 调路由，断言 SSE 中事件分隔和顺序；第二轮假模型收到首轮 human/AI 消息；生成中抛错时收到 `error` 且未保存本轮；空消息为 422；未知会话 ID 在开始流前为 404。
- [ ] **Step 2: 运行 `uv run --no-sync pytest tests/test_chat_stream.py -q`，确认接口缺失导致失败。**
- [ ] **Step 3: 实现最小代码。** 使用 FastAPI `StreamingResponse(media_type="text/event-stream")` 包装异步生成器；SSE 数据用 `json.dumps(..., ensure_ascii=False)` 编码，写成 `event: name\ndata: json\n\n`；`session` 先发送 ID，随后将 `ChatOpenAI.astream` 的文本 chunk 原样逐块发送，不额外拼词；完整成功后才 `append_turn`。流内异常发 `error`，不返回堆栈或密钥。
- [ ] **Step 4: 运行 `uv run --no-sync pytest tests/test_chat_stream.py -q`，确认通过；立即追记任务结果。**

## Task 4: 售后结构化提取与标注验证

**Files:** Create `src/customer_service/aftersales.py`, `tests/test_aftersales.py`, `evals/aftersales.jsonl`; Modify `src/customer_service/main.py`.

**Interfaces:** `AfterSalesInfo` 含 `order_id: str | None`、`request_type: Literal["refund", "return", "exchange", "repair", "logistics", "other"]`、`expected_resolution: str | None`；`ExtractionService.extract(description: str) -> AfterSalesInfo`；`POST /aftersales/extract` 返回该模型 JSON。

- [ ] **Step 1: 先写 6 条标注样例。** 覆盖有订单号退款、无订单号换货、物流催促、维修、未说明方案和无法归类。测试用假 `with_structured_output` 模型跑全部样例，并断言输出和标注一致；另测无效类别、空模型回复给出明确错误，空描述为 422。
- [ ] **Step 2: 运行 `uv run --no-sync pytest tests/test_aftersales.py -q`，确认接口缺失导致失败。**
- [ ] **Step 3: 实现最小代码。** 调用 `model.with_structured_output(AfterSalesInfo, method=settings.structured_output_method)`，将 `EXTRACTION_PROMPT` 格式化后的消息传入 `ainvoke`；对返回值用 Pydantic 校验；解析失败转为明确 502，不吞错。若 `json_mode`，Prompt 已明确要求 JSON。
- [ ] **Step 4: 运行 `uv run --no-sync pytest tests/test_aftersales.py -q`，确认标注样例全通过；立即追记任务结果。**

## Task 5: 端到端收口与交付

**Files:** Create `README.md`; Modify `dev-notes/ch01.md`.

- [ ] **Step 1: README 写出 Windows 环境准备、复制 `.env.example` 为 `.env` 后填写真实配置、`uv run uvicorn customer_service.main:app --app-dir src --reload` 启动命令。** 写出两次 `curl.exe -N` 聊天请求（第二次复用首轮 `session` ID）与一次售后提取请求；说明 `STRUCTURED_OUTPUT_METHOD` 按模型能力设置。
- [ ] **Step 2: 运行 `uv run --no-sync pytest -q`、一次本地 SSE/JSON 冒烟验证；若用户提供可用 `.env`，再用真实 GLM 或 DeepSeek 跑三项 curl。** 分别记录离线通过项、真实上游结果及未覆盖的另一家模型。
- [ ] **Step 3: 依 `superpowers:requesting-code-review` 完成审查，修复发现的问题并重测；立即追记 code review 结论。**
- [ ] **Step 4: 依 `superpowers:verification-before-completion` 核对最后一次测试输出；追记 finish 阶段，交付演示命令、测试结果和 dev-notes 路径。**

## 计划自查

- 覆盖：SSE、多轮、PromptTemplate、结构化提取、token 裁剪、配置、错误、标注样例、curl 验收、开发记录均有对应任务。
- 无工具调用、Agent 循环、页面或持久化开发。
- 关键外部边界：GLM/DeepSeek 具体模型的 `with_structured_output` 模式兼容性和真实 SSE，必须在提供有效上游配置后验证；离线测试不替代该结论。
