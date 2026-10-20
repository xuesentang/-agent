# CH05 · Workflow 编排 + 主力 Agent 实施计划

日期:2026-10-20 | 依据 `docs/superpowers/specs/2026-10-20-ch05-workflow-agent-design.md`

| 任务 | 内容 | 验收 |
|---|---|---|
| T1 | 裸循环 `app/agent/naive_loop.py`:while 循环 + 停止条件 + token 计数 + CLI | `tests/test_naive_loop.py`:无工具一步收敛 / 有工具执行后收敛 / 撞步数上限 / 撞预算上限 四条通过 |
| T2 | 意图识别 prompt(七类 JSON)+ `INTENT_ROUTES` 分流表 + 解析兜底 | `tests/test_intent.py` 单测 + `tests/eval/ch05_intent_cases.jsonl` 标注样例跑真实准确率 ≥90% |
| T3 | 置信度闸 `app/agent/gate.py`:分数粗筛 + 自评(fail-closed)+ 入池原因分级 | `tests/test_gate.py` 六条:分数过闸/分数挡下/自评判弱/降级判弱(reason 带 degraded)/业务类不过闸/入池 |
| T4 | 主力 Agent 节点:ReAct 循环 + 步数/token 上限 + 缺信息追问 + 工具事件旁路 | `tests/test_agent_node.py`:一次调用收敛 / 两步(查单→查物流) / 撞上限收尾 / 工具失败回灌 |
| T5 | 图组装 `app/agent/graph.py`:7 节点 + 条件边 + checkpointer 工厂 | `tests/test_agent_graph.py`:四出口各走对 / 知识类必过检索与闸 / 同一 thread 状态留存 |
| T6 | 服务层 `app/services/agent_chat.py`:图 → SSE 适配(token/tool/citations/actions/done/node) | `tests/test_agent_chat_service.py`:事件序、落库轨迹、拒答路径、actions 下发 |
| T7 | 路由切换 + 两个动作接口(transfer / create-ticket) | `tests/test_routes.py` 新增:动作接口写库/不写库、参数校验 |
| T8 | 前端:actions 两按钮 + 转人工模拟 + 建工单回执;顺带手测 CH04 角标与反馈 | 浏览器五条验收全走一遍,截图/curl 记录进 dev-notes |
| T9 | eval:`tests/eval/ch05_intent_cases.jsonl` + `run_intent_eval.py`,真实模型准确率 | ≥90%,错误样本逐条记录 |
| T10 | code review + README + 能力差异 + 提交推送 | 章验收 5/5 |

## 验收标准对照(提示词第五章)

1. 政策类问题日志里能看到强制检索节点 → T5 图边 + T6 `node` 事件 + T12 结构化日志;
2. 「订单 1001 的物流到哪了」Agent 自己调工具 → T4 ReAct + T6 工具事件;
3. 「我要投诉」出两个独立按钮,分别点、互不绑定 → T8 前端 + T7 接口;
4. 闲聊固定话术 → T5 `chitchat_reply` 节点(零模型调用);
5. 「先查订单再查物流」ReAct 走多步 → T4 两步用例 + T6 events 里两个 tool_start。

## 风险

- **custom stream 的 chunk 形态随版本变**(元组 vs 字典):适配层两种都认,并有单测锁死。
- **自评 fail-closed 会让 knowledge 类问题在上游抖动时整批拒答**:用 degraded 标记区分「判了弱」与「没判成」,日志可分辨;这是 CH04 review 明确要求的方向,不是回退。
- **图里拿不到 DB session**:节点不直接开库,检索用的 `session_factory` 由服务层注入 `Runtime.context` 或闭包;落库仍在服务层,节点保持纯函数(可单独单测)。
