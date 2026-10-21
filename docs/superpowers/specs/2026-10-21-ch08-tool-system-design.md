# CH08 工具系统设计

> 需求原文:`各章提示词.md` 476-540 行(功能 7 点 + 技术栈 + 5 条验收)。

## 问题

CH02 的工具层是「几个函数 + 一张注册表 + 一个执行器」,能跑但缺三样:接入只能改代码、调错了没人知道、写操作没有闸。本章把这层升级成工具系统。

## 决策

### D1 工具元数据走 sidecar,不做 `BaseTool` 子类

内置工具和 MCP 工具都得带「来源 / 是否写操作 / 来自哪个 Server」。塞进 `BaseTool` 要么继承要么改 `args_schema`,两条路都会让「LangChain 工具」和「我们的工具」绑死。

做法:`ToolRegistry` 内部存 `ToolEntry = (tool, meta)`,`meta` 是 `ToolMeta(name, source, read_only, server)`。取工具给模型时只交 `BaseTool`(协议不变),引擎侧查 meta 做权限与审计。**新增一类工具不改协议,只多一个 meta。**

### D2 写操作的门在引擎外:引擎只判定,不确认

「没确认的调用引擎直接拒绝」——但确认要弹前端、要暂停图,引擎(ececutor)是纯异步函数,给不了这点。

分工:
- `ToolExecutor` 只回答「这个工具是不是写操作」(`meta.read_only is False`),并接受一个 `confirmed: bool`;
- 真正的闸在 graph 的 agent 节点:发现写调用 → `interrupt()` 把工单预览推给前端 → resume 回来带 `approve/reject`;
- 引擎拿到 `confirmed=False` 的写调用 → 直接返回「权限拒绝」并落审计,不给模型绕过路径。

**为什么闸放节点而不是引擎**:interrupt 只有图内能用;引擎被单测、脚本、未来别的图复用,不能依赖图。

### D3 MCP 工具走「拉一次 + TTL 缓存」,不驻留 session

`MultiServerMCPClient.get_tools()` 每次调用会为每个 Server 建临时 session 拉清单,工具调用时再建 session。所以「Server 重启加了工具、客服系统不重启也能用上」= 定时重新拉一次清单。

实现:`McpToolSource.refresh(force=False)`,TTL 30 秒;每轮 chat 开始调一次(缓存没过期就返回上次结果)。拉取失败**保留上次清单**并记 WARNING——MCP Server 挂了不该让整个客服瘫痪。

### D4 工具名冲突:内置让位前先下线,不靠改名

需求:物流查询由 MCP Server 接管、内置 `query_logistics` 从清单下线、别留重名。
做法:配置 `MCP_ENABLED=true` 时,注册表跳过内置 `query_logistics`。注册表保留「重名即报错」的硬约束——重名说明配置错了,静默覆盖比报错更危险。

### D5 审计写失败不许反拦执行

审计是旁路。`record()` 内部 try/except 全吞,只记 ERROR 日志。表不挂外键(DDL 已定),`conversation_id` 从 contextvar 取,没有就 NULL。

**顺序**:先执行、后落审计。反过来的话审计表一慢,所有工具都跟着慢。

### D6 重试只看「暂时性」,写操作恒不重试

- 只读工具 + 异常属于 `transient_errors` → 重试 1 次;
- 写操作(`read_only=False`)→ 永不重试。超时未必没执行,重复执行比失败更糟;
- 业务空结果(`ok=True` 的「未找到」)→ 不重试,重试也还是空。

### D7 结果格式化收在引擎一处

工具返回 `str | dict | list`。引擎统一:`dict/list → json.dumps(ensure_ascii=False)`;超过 `TOOL_RESULT_MAX_TOKENS` 截断并注明;`None` → 空串。

需求说的「只挑用得上的字段、枚举翻人话」放在**工具自己**做(工具知道哪个字段有用);引擎只保证「结构化数据一定变成人话可读的 JSON 文本,中文不转义」。引擎替工具挑字段等于猜业务语义。

### D8 两个 MCP Server 各起一个进程,数据源复用内置 mock

`scripts/mcp_logistics_server.py`(端口 8121):`query_logistics_track`(查轨迹)。
`scripts/mcp_aftersale_server.py`(端口 8122):`check_warranty`(查在保)、`query_return_progress`(查退货进度)。

Server 内部数据用与内置工具同一份随机生成规则(同入参结果确定),不建表、不接真实系统。**工具名与内置不同名**,避免「下线哪个」的歧义。

### D9 工单确认流复用 CH06 的数据通道

CH06 订单选择器是「下发 `order_options` 事件 + 前端点选后重发」,没引入 resume 端点。本章需求明确要求 **interrupt + resume**,所以新增:

- `POST /api/v1/chat/resume`,`{conversation_id, decision: approve|reject}`;
- 前端收到 `interrupt` 事件(`kind=ticket_preview`,带类型与描述)→ 渲染预览卡片 → 点按钮调 resume → 后端 `Command(resume=...)` 继续图,SSE 继续吐 token。

**两种确认方式并存**:CH05 投诉流程的「建工单」按钮是「点按钮即确认」,保持不动(需求第 7 点明说)。

### D10 SSE 事件只增不改

新增:`tool_audit`(可选,调试用,默认不推)、`interrupt`(工单预览)。已有事件契约不动。

## 不做

- Skill 机制(需求明说只是生态概念);
- 更多外部系统接入;
- MCP Server 的鉴权与多租户(本地进程,没有边界要守)。

## 风险

- `interrupt` + `astream(stream_mode="custom")` 组合:resume 时 graph 要从 checkpoint 重放,当前服务层是「每轮新建 state 输入」,resume 必须不带输入、只带 `Command(resume=...)`;
- MCP 客户端与 FastAPI 事件循环共用:session 在 `get_tools()` 里临时建,不跨轮持有,避开「session 绑在已关闭的循环上」;
- 装 `langchain-mcp-adapters` 把 `langchain-core` 从 1.2.17 带到了 1.6.12,已跑全量 378 条确认无回归。
