# CH08 实施计划

前置:spec `docs/superpowers/specs/2026-10-21-ch08-tool-system-design.md`(D1-D10)。

## 任务

| # | 任务 | 产出 | 验收 |
|---|---|---|---|
| T1 | 工具元数据与注册中心 | `app/core/tools/registry.py`:`ToolMeta`、`register(tool, meta)`、`meta(name)`、`specs()`;内置工具补 meta | `tests/test_tool_registry.py` |
| T2 | 审计 | `app/models/tool_audit.py`、`app/repositories/audit_repo.py`、引擎每次调用落一条(含拦截)、写失败不反拦 | `tests/test_tool_audit.py` |
| T3 | 权限 + 重试规矩 + 结果格式化 | 引擎:`read_only=False` 且未确认 → 权限拒绝;写操作不重试;`_format_result` | `tests/test_tool_permission.py` |
| T4 | MCP 接入 | `app/core/tools/mcp_source.py`(TTL 缓存 + 失败保留)、`config` 增配置、`build_default_registry` 合并 | `tests/test_mcp_source.py`(假 client) |
| T5 | 两个 MCP Server | `scripts/mcp_logistics_server.py`(8121)、`scripts/mcp_aftersale_server.py`(8122) | 起进程 + curl 握手 + 客户端拉到工具 |
| T6 | 工单确认流 | graph agent 节点 interrupt;`POST /chat/resume`;前端预览卡片 | `tests/test_ticket_confirm.py` + 真链路 |
| T7 | 文档与验收 | README、dev-notes、能力差异、`scripts/ch08_demo.py` | 5 条验收 |

## 顺序理由

先元数据(T1)再引擎(T2/T3)——审计和权限都要读 meta;MCP 接入(T4)只要元数据接口稳定就能接;Server(T5)独立进程、可以最后起;确认流(T6)动图,放最后以免中间状态干扰前面测试。

## 风险

- T6 会改 agent 节点与 SSE 契约,既有 378 条里凡涉及 agent 节点的都要核;
- T4 的 `get_tools()` 是网络调用,单测必须注入假 client,不许连真 Server。
