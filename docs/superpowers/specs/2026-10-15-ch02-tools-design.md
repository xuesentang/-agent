# CH02 Function Calling 工具链 — 设计 Spec

> 状态：已定稿待执行（单章连续执行模式，spec/plan 落盘即自查通过，不等阶段批准；仅遇「重要问题 / 难回退选型 / 需用户外部资源」停下）。
> 依据：`各章提示词.md` 第二章、`各章建表.md` ch02 DDL、`AGENTS.md`。

## 1. 目标与范围

给 CH01 的纯对话客服装上查数据能力：模型通过 Function Calling 自己决定调什么工具，工具链长在 SSE 流式聊天里，用户问一句即触发「模型定工具 → 执行 → 回灌收敛」，最终回答逐 token 流出，工具执行段推状态帧，全量聊天记录（含工具调用与结果）落 MySQL，前端气泡显示工具徽章。

**本章不做**：多轮自动 Agent Loop（CH05）、向量检索 / RAG（CH03+）、权限与审计的正式版（CH08，本章 create_ticket 为演示直写）。

## 2. 决策点

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | 数据库 | Docker MySQL 8，主机端口 **3307**（避让本机 3306 占用），`DATABASE_URL` 走 `.env`（`mysql+aiomysql://...`），docker-compose.yml 交付 | 提示词定死 Docker MySQL；端口避让不改方案只改映射；URL 配置化保证本机 / Docker 可切 |
| D2 | 驱动 | 生产 **aiomysql**，测试 **aiosqlite**（同步 pymysql 已装但弃用于主路径） | 全异步栈一致（CH01 即 async 路由 + astream）；SQLAlchemy 2.0 AsyncEngine 成熟；pymysql 留作 CLI / 脚本导入 |
| D3 | 分层 | `api/routes` → `services` → `repositories` → `models`（SQLAlchemy 2.0 ORM `Mapped`/`mapped_column`）；`core/` 放 llm/prompts/session/tools/schemas | CH01 已有 core/services/api 三层，repositories/models 为数据访问新增；单向依赖不变 |
| D4 | 单轮工具调用 | `llm.bind_tools(tools)` → `invoke` 拿 tool_calls → 推 SSE 状态帧 → 注册中心调度执行 → `ToolMessage(tool_call_id)` 回灌 → `astream` 收敛逐 token | 提示词「只做单轮调用」；官方工具循环模式（Context7 查证）截到一轮；状态帧满足「工具执行那段先推状态」 |
| D5 | 五工具契约 | 见 §4 | 提示词逐条 |
| D6 | 消息落库 | user/assistant/tool 全量落 messages；assistant 的 tool_calls 存 `tool_calls` JSON 列；tool 消息 content 存结果摘要、tool_call_id 对号 | 建表 DDL 定义；回灌历史时可原样重建 |
| D7 | SSE 协议扩展 | 新增 `tool_start`（name/args）、`tool_end`（name/ok/summary）；`token`/`done`/`error` 不变 | CH01 协议向后兼容 |
| D8 | 测试策略 | 单测跑 **SQLite 内存库**（aiosqlite，方言差异限 CRUD+LIKE 无风险）；集成验收跑 **Docker MySQL 真库 + 真实 GLM**；不可单测项（Prompt 选工具准确率）用标注样例 eval ≥90% | 单测不依赖外部服务保持快速；验收必须真栈（ch01 教训：通道行为只有真调才知道） |
| D9 | 种子数据 | `scripts/seed_faq.py` 等脚本 AI 生成逼真数据（faq 多分类、对话式问法、含 LIKE 漏召回样例「邮费/运费」）；conversations/messages/tickets 少量演示种子；注明 AI 合成 | AGENTS.md 第 2 节；脚本化可复现 |
| D10 | 前端聊天页 | 原生 HTML+CSS+JS 静态页（FastAPI StaticFiles），EventSource 消费 SSE，渲染 token / tool 状态帧 / 工具徽章；走 Vibe Coding 例外（不套 TDD/review） | 提示词验收 1 要求浏览器打开；CH05 明确原生三件套 |

## 3. 架构与数据流

```
浏览器聊天页 ──POST /api/v1/chat/stream(SSE)──▶ routes.chat_stream
   │                                                │
   │  EventSource                                  ▼
   │                                         services.chat_with_tools
   │                                         │ 1. 读/建会话(conversations)
   │                                         │ 2. 组装历史(messages, 复用 CH01 trim)
   │                                         │ 3. llm.bind_tools(五个工具).invoke
   │                                         │ 4. 有 tool_calls?
   │                                         │      yield tool_start 状态帧
   │                                         │      注册中心调度:
   │                                         │        Schema 校验 → 执行 → 超时重试 → 格式化
   │                                         │      yield tool_end 状态帧
   │                                         │      ToolMessage 回灌
   │                                         │ 5. astream 收敛, yield token…
   │                                         │ 6. user/assistant/tool 全量落 messages
   ▼                                         ▼
 SSE 事件: tool_start / tool_end / token / done / error       repositories(SQLAlchemy AsyncSession)
```

单轮收敛定义：模型第一次 invoke 若产生 tool_calls，执行并回灌后**再调用一次**（仍 bind_tools，允许模型这次直接作答）；若第二次仍产生 tool_calls（GLM 偶发），直接以工具结果摘要组织兜底答复，不再执行——保证「只做单轮调用」且不无限循环。

## 4. 五个工具契约

| 工具 | 来源 | 入参 | 行为 | 返回 |
|---|---|---|---|---|
| `query_order` | mock | `order_id: str` | 内部按 order_id 确定性随机生成（seed=hash(order_id)，保证同订单同结果） | 订单号/商品/金额/状态/下单时间 |
| `query_product` | mock | `product_name: str` | 同上 | 商品名/价格/库存/规格 |
| `query_logistics` | mock | `order_id: str` | 同上 | 轨迹列表（时间+地点+状态） |
| `query_faq` | DB | `keyword: str` | `SELECT … WHERE question LIKE %kw% OR answer LIKE %kw% ORDER BY id LIMIT n` | 命中行 or 「未找到相关问题」 |
| `create_ticket` | DB | `description: str, ticket_type: 售后\|投诉\|咨询` | 生成工单号 `T+年月日+4位序列`，写 tickets 表 | 工单号+确认语 |

- mock 三工具**不建表不接真实接口**，函数内 `random.Random(seed)` 生成，可复现（AGENTS.md 逼真标准：金额量级、物流节点城市、时间线合理）。
- 参数 Schema：每个 @tool 的类型注解即 schema；注册中心在执行前用 pydantic 校验（`tool.args_schema`），校验失败包成工具结果回灌不抛异常。
- 错误分诊（CH08 正式版的前身）：参数不合法 → 校验拦下回灌；查询落空 → 空结果结果回灌；执行异常 → 错误信息回灌，SSE 不受影响。

## 5. 数据库（ch02 DDL，四张表）

按 `各章建表.md` ch02 节执行：`conversations`（会话壳）、`messages`（流水，tool_calls JSON）、`faq`（问答对）、`tickets`（工单）。建表脚本落 `scripts/sql/ch02_schema.sql`（与建表语句逐字一致，utf8mb4）；ORM 模型与 DDL 字段一一对应。CH03+ 的 knowledge_chunks 等表不在本章。

## 6. 测试策略

- **单测（SQLite）**：模型层 CRUD、仓库层（faq LIKE 命中/未命中/注入字符）、工具层（mock 确定性、create_ticket 工单号格式）、服务层（mock LLM：单轮 tool_calls 收敛 / 无 tool_calls 直接答 / 工具异常回灌 / 第二次仍要工具的兜底）、SSE 协议（tool_start/tool_end/token/done 事件序列）、消息落库（tool_calls JSON 对号）。
- **eval（真实 GLM）**：`tests/eval/ch02_tool_cases.jsonl` 12-15 条标注样例——问句 → 期望工具名（query_logistics/query_faq/create_ticket/无工具），通过率 ≥90%。
- **集成验收（Docker MySQL + 真实 GLM）**：提示词三条验收标准现场跑。

## 7. 验收标准映射

1. 浏览器问「订单 1001 的物流到哪了」→ 徽章显示 query_logistics + 按结果作答 → 集成验收 curl/浏览器 + eval 样例。
2. 问「退货政策是什么」→ query_faq 命中作答 → 集成验收（种子数据含该条目）。
3. 问「邮费是多少」→ LIKE 查不出（种子数据用「运费」不用「邮费」）→ **预期漏召回，记入 README 已知限制，留给 CH03 向量检索升级**。

## 8. Review Focus

1. 未知 conversation_id → 静默新建（CH01 session 语义保持）。
2. tool_calls 回灌历史重建：assistant(tool_calls) + tool(tool_call_id) 两条消息顺序与 id 对号。
3. create_ticket 工单号并发唯一（序列取 max+1 或日期+随机段，重试不撞）。
4. LIKE 特殊字符（% _ \）转义，防注入与误匹配。
5. SSE 工具状态帧与 token 帧交错时序：tool_start → token… → tool_end 不得乱序。
6. mock 工具确定性：同入参两次调用结果一致（seed 派生）。
