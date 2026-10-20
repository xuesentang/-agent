# CH05 · Workflow 编排 + 主力 Agent 设计

日期:2026-10-20 | 范围:第五章提示词全部功能需求

## 决策点

**D1 骨架用 LangGraph,主力 Agent 自己写循环,不用 prebuilt**
`create_react_agent` 藏掉了停止条件与 token 计数的位置,而本章的验收点恰恰是「能看见 ReAct 走了不止一步」「要有停止条件和 token 控制」。所以 `agent` 节点内部手写循环:`bind_tools.ainvoke` → 有 tool_calls 就经 `ToolExecutor` 执行 → 回灌 → 再调,直到无 tool_calls 或撞到步数/预算上限。图负责确定性骨架,循环负责不确定性推理,分工清晰。

**D2 祛魅 demo 先落地,再重构**
`app/agent/naive_loop.py`:没有图、没有节点、没有状态,就是一个 while:调模型 → 有工具就执行 → 把结果塞回去 → 没有就停。可 CLI 直接跑,也可单测(假模型断言步数)。图版本落地后它仍在,作为「同一件事的两种写法」对照物。

**D3 State 只装本轮需要的东西,历史仍以 MySQL 为准**
`AlgoState`:user_input / resolved_query / intent / confidence / route / evidence / gate / reply / finish_reason / actions / trace / steps / tokens。历史由服务层从 `messages` 表读出、裁剪后随输入注入 `history` 字段(每轮覆盖,不用 reducer)。理由:CH03/04 的落库与配对清洗是验证过的,换 State 持有全量历史等于把已验证的东西重做一遍;而 CH07 要做的正是「State 里挂 add_messages + 摘要压缩」,那时再换。

**D4 checkpointer 按 thread_id = conversation_id 落地,优先 SQLite,拿不到就内存**
"持久化用自带 checkpointer":`langgraph.checkpoint.memory.InMemorySaver` 是自带的内存版,进程重启即丢。装了 `langgraph-checkpoint-sqlite` 就自动切 `AsyncSqliteSaver`(落 `./data/agent_checkpoints.db`),没装则降级并在启动日志里说明。**不为一章的功能往用户环境装依赖**(装包是红线项),用「有就更好、没有也能跑」的工厂规避。

**D5 意图识别:一个 prompt 输出七分类 JSON,解析失败落到「售后」**
七类写死在 `INTENT_ROUTES` 表里;`confidence` 一并返回但本章只记录、不做升降级(CH06 的降级路才用)。解析失败/模型异常 → 归「售后」(业务数据类,直进 Agent,不预检索),并置 `intent_fallback=True` 留痕。为什么不落到「闲聊」:闲聊零模型调用,把解析失败的问题当寒暄是最糟的失败方向。

**D6 分流规则是一张表,不是 if-else 森林**
`INTENT_ROUTES = {物流: business, 订单: business, 商品咨询: knowledge, 退款退货: knowledge, 售后: business, 投诉: complaint, 闲聊: chitchat}`,四个出口 `knowledge / business / complaint / chitchat`。表驱动的好处:CH06 加「其他」类只加一行,改分流只改一格。

**D7 置信度闸 = 分数粗筛 + 证据自评,位置在检索之后、Agent 之前**
`top1_score < KB_MIN_SCORE(0.45)` → 直接判弱,不花自评的 token;否则跑 CH04 的 `check_evidence`。**自评异常改 fail-closed**(CH04 review 留下的待办,本章兑现):有了「转人工/建工单」出口,多拒一次的代价从「服务停摆」降到「多点一下」。落地方式是给 `check_evidence` 加 `on_error="pass"|"block"` 参数,默认 "pass" 保持旧行为、旧测试不动;闸这边显式传 "block"——策略写在调用点上,不藏在被调函数里。

**D8 拒答兜底要带出口**
闸判弱 → 回 `ABSTAIN_REPLY`(固定话术)+ `actions=[transfer_human, create_ticket]`,并写 `low_confidence_questions`(reason 里区分 `score_low` / `self_check` / `degraded`)。拒答但不给下一步,等于把用户晾在原地。

**D9 投投诉/闲聊零模型调用**
投诉走安抚话术 + 两个可选项;闲聊走固定话术。两条都在图里,不发一次 LLM 请求——验收标准 4 的前提,也是「确定性编排」的价值:能确定的事不交给模型。

**D10 转人工 / 建工单是两个独立动作,后端不自动执行**
`done` 事件带 `actions` 数组,前端各渲染一个按钮,互不绑定:
- 转人工:`POST /api/v1/chat/actions/transfer` —— 纯前端模拟接真人,后端只落一条 `assistant` 消息留痕并返回问候语(可测、可回放);
- 建工单:`POST /api/v1/chat/actions/create-ticket` —— 走 CH02 的 `create_ticket` 工具(经注册中心 + 执行引擎),不直接写 repo,保证「工具」这条路径唯一。

**D11 SSE 事件契约向后兼容,只增不改**
`token` / `tool_start` / `tool_end` / `citations` / `done` / `error` 原样保留(`done` 增 `actions`、`intent`、`route`);新增 `node` 事件(节点进入/离开),让前端与 curl 都能看见「强制检索节点被走到」——验收标准 1 靠它,不靠翻服务端日志。

**D12 日志节点产出结构化一行,不建表**
第五章建表语句是空的,不新增表(改 schema 属红线)。`finalize` 节点打一行 `agent.trace` JSON:conversation_id / intent / route / gate / steps / tokens / finish_reason / 耗时。管线外部可 grep,面试能讲清「日志落在哪、为什么不为它建表」。

## 本章不做

- 指代消解、Query 改写/扩写的正式版(CH06);
- 上下文压缩与摘要(CH07);
- 置信度闸的正式版(可观测章);
- 转人工接真人系统(明确只做前端模拟)。
