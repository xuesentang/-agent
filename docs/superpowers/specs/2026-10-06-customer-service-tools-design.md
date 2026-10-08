# 第二章：客服聊天工具调用设计

## 目标与边界

在现有 `/chat/stream` SSE 入口加入一次模型工具决策、一次工具执行和最终回答流式输出。用户仍在现有聊天页提问；页内只显示本轮调用过的工具徽章。继续使用 DeepSeek 的 OpenAI 兼容接口、FastAPI 和 LangChain。本章不做多轮 Agent Loop、RAG、真实电商/物流 API，也不新增订单、商品、物流表。

## 运行环境与数据

使用本机已安装的 MySQL80（`127.0.0.1:3306`），不依赖 Docker。项目使用独立的 `customer_service` 数据库，不修改现有其他库。数据库连接从 `.env` 的 `DATABASE_URL` 读取，密码不进入代码、日志或示例。启动脚本显式创建四张表并幂等灌入 FAQ 样例；不在普通请求中建表，不执行破坏性迁移。此段为 2026-10-06 用户确认的环境方案修订，替代初稿的 Docker MySQL 设计。

四张表：

| 表 | 字段 |
| --- | --- |
| `faq` | `id` 主键，`question`、`answer`、`category` |
| `conversations` | `id` 主键，`user_id`（可选输入，默认 `guest`）、`status`、`created_at` |
| `messages` | `id` 主键，`conversation_id` 外键，`role`（`user`/`assistant`/`tool`），`content`，`tool_calls` JSON 可空，`tool_call_id` 可空，`created_at` |
| `tickets` | `ticket_id` 工单号主键，`conversation_id` 外键，`description`、`ticket_type`、`status`、`created_at` |

`conversations.id` 沿用字符串 UUID 形式，浏览器现有 `conversation_id` 不变。`messages` 按时间和递增主键重建历史；一次有工具的完整轮次依次保存 user、含调用申请的 assistant、tool 结果、最终 assistant。无工具则保存 user 和最终 assistant。仅完整完成的最终答复记入历史；若流中断，已发生的调用/结果仍留痕，但不伪造完成的最终答复。服务端保持单进程演示，数据库负责持久化。

## 五个业务工具

用 LangChain `@tool` 和 Pydantic `args_schema` 定义并注册：

- `query_order(order_id)`：内部生成演示订单状态，不调用真实订单 API。
- `query_product(product_name)`：内部生成演示商品信息，不调用真实商品 API。
- `query_logistics(order_id)`：内部生成演示物流节点，不调用真实物流 API。为了本轮回答可信，同一次调用生成的结果原样回灌和持久化，模型不得自行补造未返回的信息。
- `query_faq(keyword)`：仅对 `faq.question` 做 SQL `LIKE` 关键词匹配，无向量检索、同义词扩展和模糊语义匹配。样例含“退货政策”，不含“邮费”；“退货政策是什么”可命中，“邮费是多少”应返回未找到。
- `create_ticket(description, ticket_type)`：将人工工单写入 `tickets`，生成唯一工单号并关联当前会话。模型只可在用户明确要求转人工或建立工单时调用。

工具注册表按名称查找，执行前用各工具的 Pydantic schema 校验参数；未知工具、参数错误、超时和执行错误转换为可回灌的 `ToolMessage`，不暴露异常堆栈。四个只读工具限定超时并最多重试一次；`create_ticket` 不自动重试，避免重复写入。模拟数据的随机性只影响演示内容，测试通过注入随机源保持可重复。

## 聊天链路与 SSE

`/chat/stream` 请求在现有 `message`、`conversation_id` 基础上允许 `user_id`。新会话写入 `conversations`，旧 ID 从数据库加载；未知 ID 返回 404。读取历史后使用现有近似 token 预算裁剪，保留工具调用申请与其结果的完整配对，避免构造非法模型消息序列。

模型以 `bind_tools([...], tool_choice="auto", parallel_tool_calls=False)` 决定是否调用工具。工具阶段不流式输出模型文字；若模型给出一个工具调用，SSE 发 `tool_status`（`tool_name`、`state=running`），执行并保存后再发 `tool_status`（`state=success` 或 `error`）。工具结果用匹配 `tool_call_id` 的 `ToolMessage` 回灌；最终回答调用**不绑定工具的模型**并用 `astream` 逐块发 `token`，最后 `done`。模型若意外返回多个工具调用，明确报错，不暗自执行部分调用。若没有调用工具，仍用无工具模型流式生成回答。本章每轮最多一次工具调用、一次最终收敛，不做 Agent Loop。

SSE 继续以 `session` 开始，`token` 和 `done` 结束。聊天页只在对应助手气泡显示实际执行工具的紧凑徽章（例如 `query_logistics`），无工具则不显示；不展示内部参数、JSON 结果或独立工具面板。页面改造按用户指定的 Vibe Coding 工作方式。

## 验收与失败边界

1. 浏览器问“订单 1001 的物流到哪了”，后端走 `query_logistics`，页面出现该工具徽章，最终回复依据工具返回的物流数据。
2. 问“退货政策是什么”，走 `query_faq`，SQL `LIKE` 命中 FAQ 样例并回答。
3. 问“邮费是多少”，工具可被调用，但 `query_faq` 的关键词 SQL 查询不命中；最终回答明确查无 FAQ，不编造邮费。记录这个预期漏召回，留待下一章升级。
4. 核对数据库中的会话、user/assistant/tool 消息及工单记录；离线单测、浏览器 SSE 测试和真实 DeepSeek 工具选择测试分别报告。工具选择受模型输出影响，验收样例需真实运行，不能只用假模型代替。

## 文档依据

- [SQLAlchemy 2.0 ORM 映射与 MySQL URL](https://docs.sqlalchemy.org/en/20/core/engines.html)：`mysql+pymysql://`；[事务管理](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html)。
- [LangChain 工具定义](https://docs.langchain.com/oss/python/langchain/tools)：`@tool(args_schema=...)`；[模型工具调用与结果回灌](https://docs.langchain.com/oss/python/langchain/models)：`bind_tools`、`AIMessage.tool_calls`、`ToolMessage`。
- [FastAPI 流式响应与数据库依赖](https://fastapi.tiangolo.com/advanced/advanced-dependencies)。
