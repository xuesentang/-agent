# CH01 纯对话跑通 — 设计文档

> 状态：待用户评审
> 日期：2026-10-15
> 作者：薛森堂

## 1. 目标与背景

电商智能客服系统第一章：**跑通纯对话**。本章交付一个 FastAPI 服务，具备 SSE 流式多轮对话、模板化客服 Prompt、售后描述结构化抽取三项能力。工具调用、Agent 循环、RAG、数据库一律不做（见 §3）。

模型接入层统一 OpenAI 协议，地址/模型名/密钥全部走 `.env`，换供应商只改配置不改代码。当前供应商定 **GLM-5.3-Flash（智谱）**——选型过程与查证记录见 `dev-notes/ch01.md`。

## 2. 范围

### 2.1 本章做

| # | 功能 | 说明 |
|---|------|------|
| F1 | SSE 流式对话 | `POST /api/v1/chat/stream`，逐 token 推送，多轮上下文 |
| F2 | Prompt 模板化 | `ChatPromptTemplate` 管理客服 system prompt（角色设定 + 行为约束） |
| F3 | 结构化抽取 | `POST /api/v1/chat/extract`，售后描述 → 订单号/诉求类型/期望方案，`with_structured_output` 实现 |
| F4 | 上下文裁剪 | `trim_messages` + token 预算，最简版 |

### 2.2 本章不做（明确边界）

- 工具调用与 Agent 循环（注：F3 的 `with_structured_output` 是 LangChain 的 schema 约束机制，不是 Agent 工具循环，属本章范围）
- 用户认证、持久化存储、会话过期清理
- RAG / 知识库
- 前端页面（本章只有 API；聊天页面后续章节单独做）

## 3. 架构

### 3.1 目录结构

```
D:/客服agent/
├── .env.example            # 配置模板（无密钥）
├── .env                    # 实际配置（用户填密钥，git 忽略）
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI app 装配、异常处理
│   ├── config.py           # pydantic-settings 读取 .env
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py       # /chat/stream、/chat/extract
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm.py          # ChatOpenAI 工厂（OpenAI 协议，.env 驱动）
│   │   ├── prompts.py      # ChatPromptTemplate（客服 system prompt）
│   │   ├── session.py      # 内存 session store + trim_messages 裁剪
│   │   └── schemas.py      # AfterSalesInfo（结构化输出 schema）
│   └── services/
│       ├── __init__.py
│       ├── chat.py         # 流式对话服务
│       └── extract.py      # 结构化抽取服务
├── tests/
│   ├── conftest.py
│   ├── test_session.py     # 单测：裁剪逻辑
│   ├── test_prompts.py     # 单测：模板渲染
│   └── eval/
│       └── after_sales_cases.jsonl   # 标注样例（替代 TDD 验证 Prompt/抽取）
└── dev-notes/
    └── ch01.md
```

### 3.2 模块职责与边界

| 模块 | 职责 | 依赖 |
|---|---|---|
| `config.py` | 读 `.env`，提供 `Settings` 单例 | pydantic-settings |
| `llm.py` | 按配置构造 `ChatOpenAI`（对话实例 temperature=0.7；抽取实例 temperature=0） | langchain-openai |
| `prompts.py` | 客服 system prompt 模板 + 抽取用 prompt 模板 | langchain-core |
| `session.py` | `SessionStore`：`dict[session_id, list[BaseMessage]]`；`get_history()` 返回裁剪后历史 | langchain-core |
| `schemas.py` | `AfterSalesInfo` Pydantic 模型（F3 的 schema） | pydantic |
| `services/chat.py` | 编排：取历史 → 拼 prompt → astream → 逐 token yield → 落库 assistant 消息 | 以上全部 |
| `services/extract.py` | 编排：`with_structured_output` → invoke → 返回 dict | llm + schemas + prompts |
| `api/routes.py` | HTTP 边界：请求校验、SSE 编码、错误转 SSE error 事件 | services |

### 3.3 数据流

**对话流（F1/F4）：**

```
请求 {session_id, message}
  → SessionStore 取该 session 历史（无则新建空列表）
  → trim_messages(历史, max_tokens=HISTORY_TOKEN_BUDGET)   ← 只裁历史，system 每次新拼
  → ChatPromptTemplate([system 客服人设, ...裁剪后历史, human 新消息])
  → model.astream(prompt_value)
  → 每个 chunk：SSE event=token, data={"content": chunk.text}
  → 流结束：user/assistant 消息追加进 SessionStore
  → SSE event=done, data={"session_id", "finish_reason"}
```

**抽取流（F3）：**

```
请求 {text}
  → model_extract.with_structured_output(AfterSalesInfo, method="function_calling")
  → invoke(prompt | text)
  → 返回结构化 dict（Pydantic 校验失败 → 502 错误体）
```

## 4. 关键设计决策

### D1 模型接入：OpenAI 协议 + .env 全配置

```bash
# .env
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_API_KEY=            # 用户自行填入
LLM_MODEL=glm-5.3-flash
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
HISTORY_TOKEN_BUDGET=2048
```

- 换 GPT/Claude/DeepSeek/Ollama 只改 `LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY`。
- 智谱 Coding Plan 用户的地址是 `https://open.bigmodel.cn/api/coding/paas/v4`，标准开放平台是 `.../api/paas/v4/`——默认写标准地址，用户按自己账号类型改。

### D2 结构化输出：method=`function_calling`，带回退链

- 默认 `function_calling`：GLM-5.3-Flash 的 Function Calling 是官方能力表第一方确认，schema 由 tool 定义承载，不占 prompt，Pydantic 自动校验。
- **回退链**：`function_calling` 失败 → `json_mode`（`response_format: json_object` 已多源确认）→ 仍失败则停下来问用户，**不自行换方案**（用户工作约定）。
- 不用默认的 `json_schema`：智谱未在 API 参数层明确承诺 OpenAI 的 `json_schema` 类型，不赌。

### D3 多轮上下文：服务端内存 session + trim_messages

- `SessionStore` 为进程内 `dict`，**重启即丢**——已知限制，写进 README。
- 裁剪用 LangChain 官方 `trim_messages(strategy="last", token_counter=count_tokens_approximately, max_tokens=HISTORY_TOKEN_BUDGET, start_on="human", end_on=("human", "ai"))`。
- 预算语义：**预算只约束历史部分**（默认 2048 token），system prompt 每次由模板新注入、不计入。system prompt 永远保留在首位。
- 不设 session 数量上限（YAGNI），已知限制写进 README。

### D4 SSE 协议

请求：`POST /api/v1/chat/stream`，body `{"session_id": "任意字符串", "message": "..."}`

响应（`text/event-stream`）：

```
event: token
data: {"content": "您好"}

event: done
data: {"session_id": "abc", "finish_reason": "stop"}
```

错误（上游 4xx/5xx、流中断、校验失败）：

```
event: error
data: {"code": "UPSTREAM_ERROR", "message": "..."}
```

抽取接口 `POST /api/v1/chat/extract`，body `{"text": "..."}`，响应 `200` + JSON：
```json
{"order_id": "20241015001", "claim_type": "refund", "expected_solution": "全额退款"}
```

### D5 AfterSalesInfo schema

```python
class AfterSalesInfo(BaseModel):
    order_id: str | None = Field(default=None, description="订单号，用户未提及则为 null")
    claim_type: Literal[
        "refund",            # 退款
        "return_refund",     # 退货退款
        "exchange",          # 换货
        "repair",            # 维修
        "compensation",      # 补偿/赔偿
        "complaint",         # 投诉
        "inquiry",           # 咨询
        "other",             # 其他
    ] | None = Field(default=None, description="诉求类型，无法判断则为 null")
    expected_solution: str | None = Field(default=None, description="用户期望的解决方案，未提及则为 null")
```

### D6 模型参数与思考模式注意事项

- 对话实例 temperature=0.7（客服需稳定），抽取实例 temperature=0（抽取需确定），max_tokens=512。
- ⚠️ GLM-5.3-Flash 的思考模式 `thinking.type` 仅支持 enabled、**不可关闭**。流式时 reasoning 内容走 `delta.reasoning_content`（LangChain 中落在 `chunk.additional_kwargs`），`chunk.text` 只含正式 content——SSE 只推 `chunk.text` 即可，不特殊处理。若实测延迟不可接受，停下来问用户。

### D7 客服 system prompt（要点，全文在 prompts.py）

角色：电商平台售后客服"小智"。行为约束：不承诺未授权的赔偿/退款额度；不查询订单状态（本章无工具）；涉及账号安全引导转人工；语气礼貌简洁；不知道的明说不知道，不编造。

## 5. 测试策略

按用户工作约定：**不可单测的任务（Prompt 质量、抽取准确率）用标注样例/评估集验证，其余照常 TDD/单测**。

| 对象 | 方法 |
|---|---|
| `SessionStore` 裁剪逻辑（预算、条数、边界） | pytest 单测 |
| Prompt 模板渲染（变量注入、消息结构） | pytest 单测 |
| 抽取准确率（F3） | 标注样例 `tests/eval/after_sales_cases.jsonl`（12-15 条：正常单诉求/多信息一句话/无订单号/口语化/错别字/多诉求），逐条比对字段，**通过率 ≥ 90% 算过**，不过停下来调 prompt 或换 method |
| 端到端 | curl 手动验收（见 §6） |

## 6. 验收标准映射

| # | 验收标准 | 验证方式 |
|---|---|---|
| 1 | curl 调对话接口能看到流式回复 | `curl -N -X POST .../chat/stream`，终端逐 token 打印 |
| 2 | 连续问两轮，第二轮续上第一轮上下文 | 同 session 先问"记住编号 A123"，再问"我刚才说的编号是什么" |
| 3 | 发一段售后描述，能拿到结构化 JSON | `curl -X POST .../chat/extract` + eval 集通过率报告 |

## 7. 已知限制（写进 README）

1. 会话存进程内存，服务重启即丢，无持久化。
2. 不设 session 数量与单 session 长度上限（预算裁剪只限 token，不限条数暴增）。
3. GLM-5.3-Flash 思考模式不可关，首 token 延迟高于纯非思考模型。
4. 无鉴权，仅限本地/内网开发用。
5. 抽取准确率依赖模型表现，边界 case 见 eval 集。

## 8. 依赖

fastapi、uvicorn[standard]、langchain、langchain-openai、pydantic、pydantic-settings、httpx、pytest、pytest-asyncio。安装时取兼容最新版并锁定进 requirements.txt。
