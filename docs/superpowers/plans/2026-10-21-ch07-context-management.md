# 第七章实施计划

前置:spec `docs/superpowers/specs/2026-10-21-ch07-context-management-design.md`(D1-D8)。

## 任务

| # | 任务 | 产出 | 验收 |
|---|---|---|---|
| T1 | 预算与估算 | `app/core/token_budget.py`(`estimate_tokens` / `ContextBudget` / `compute_budget`)、`config.py` 五个新配置、启动自检 | `tests/test_token_budget.py`:中文口径、倒推公式、预算不足告警 |
| T2 | 数据层 | `app/models/conversation_summary.py`、conversations 加两个锚点、`conversation_summary_repo`、`scripts/sql/ch07_schema.sql` | `tests/test_conversation_summary_repo.py` |
| T3 | 三层切分与拼装 | `app/core/context.py`:按锚点切三层、层 2 截短、拼装顺序、`model_ctx`/`history_ctx` 渲染 | `tests/test_context_layers.py` |
| T4 | 后台摘要 | `app/services/summary_service.py`:触发判定(按用量)、后台任务、落库并推锚点、全程留痕 | `tests/test_summary_service.py` |
| T5 | 接进图与服务 | `AgentState.messages`(add_messages)、喂模型的精简版、两处 ctx 日志、预算注入 | `tests/test_context_wiring.py` + 既有 328 条不回归 |
| T6 | 只读接口 + 侧栏 | `GET /api/conversations`、`GET /api/conversations/{id}/messages`;前端侧栏 | `tests/test_routes.py` 增补 + 浏览器/curl |
| T7 | 验收脚本 | `scripts/context_demo.py`:20 轮对话、演示配置下一轮级联、断言 | 手动跑,结论进 dev-notes |

## 顺序理由

先做纯函数(T1、T3)再做有状态的东西(T2、T4),最后接图(T5)。三层切分是纯函数,可以拿构造数据把边界(id 对不齐、空层、只有系统消息)全测掉,不用碰数据库和模型。摘要服务依赖预算和切分,放后面。

## 风险

- **T5 会动既有链路**:`messages` 加 reducer 后,服务层落库的取值来源要从 `trail` 改成 `messages`,既有 8 条服务层测试会有一批要改。先跑一遍基线,改完逐个核对预期变化,不许「测试跟着代码改」而不说明。
- **摘要任务与请求共用 session**:后台任务必须自己开 session(`session_factory`),不能借用请求的 session,否则请求结束后 session 关闭,任务抛异常。
