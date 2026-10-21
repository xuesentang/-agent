# 第七章设计:会话上下文管理（三层 + 异步摘要 + 预算倒推）

需求原文:`各章提示词.md` 411-474 行。5 条验收。

## 一句话

把「留最近 N 条消息」升级成「三层 + 摘要 + 从窗口倒推预算」,并让每轮真正发给模型的上下文可见。

## 决策

**D1 三层怎么切:两个锚点,不搬数据。**
`conversations.summary_upto_msg_id` = 摘要覆盖到哪条,`conversations.layer1_from_msg_id` = 层 1 从哪条起。
降级只挪这两个 id,消息行本身永远不动——搬数据会让「历史原文」在库里被改写,前端回载旧会话就看不到原样了。

**D2 层 2 的截短规则:按角色区别对待。**
- 用户原话:不动。用户说的话是判据的来源(订单号、诉求),压了就没法回溯;
- 客服答复:只留开头 `LAYER2_ASSISTANT_HEAD`(默认 40)字,加省略号;
- 工具结果:换成一行标识 `[工具 query_order 结果 218 字]`。工具结果本来就不落消息表(需求 2 的原话),层 2 里只可能来自 State 内的 trail。

**D3 摘要:后台任务、只追加、不回头重写。**
触发看**用量**(层 2 token 超 `layer2` 预算),不数条数——工具结果不落消息表,数条数看不见它涨。
摘要一段段追加进 `conversation_summaries`(每段记 `from_msg_id`/`to_msg_id`);旧梗概只作背景喂模型、不参与合并,避免同一个事实被反复有损压缩。
压完把 `summary_upto_msg_id` 与 `layer1_from_msg_id` 都推到层 1 起点。
**不阻塞**:`asyncio.create_task`,失败只记日志,不改本轮回复。

**D4 预算从窗口倒推,不写死。**
```
history = window
        - system(实测 system prompt token)
        - evidence(rerank_top_k × 单块估值)
        - summary(摘要段预留)
        - output(max_output_tokens)
        - user_input(max_user_input_tokens)
        - react_peak(max_agent_steps ×(tool_result_max + 单步正文估值))
        - safety(窗口 5%)
layer1 = floor(history × 0.7);layer2 = history - layer1
```
`history <= 0` → 打 WARNING「上下文预算不足」,本轮退回只带当前这一句(不崩)。启动时自检并打一行。

**D5 token 估算自己写,不用 `count_tokens_approximately`。**
实测:`build_agent_system_prompt` 1061 个汉字 → `count_tokens_approximately` 报 5305,五倍。它按英文 BPE 口径数中文,拿它做预算等于把窗口砍掉五分之四。
改用 `estimate_tokens`:汉字/全角标点按 1 token、其余按 4 字符 1 token,向上取整。口径与预算一起标定(需求 4 的原话),实测比值记进 dev-notes。

**D6 拼装顺序固定,梗概和证据不占 system。**
`system` → 层 2 → 层 1 → 当前这句话 → (梗概 + 证据) 合成的一条 human。
梗概单独占 system 会被上游把所有 system 上提合并,工具定义被挤到可变内容之后,前缀缓存整段作废。

**D7 State 用 `add_messages` 装全量,喂模型的是另一份。**
`AgentState.messages: Annotated[list, add_messages]`,节点只吐新消息;完整历史靠 checkpoint,喂模型的精简版每轮现算。两者互不影响(需求 5)。
重启后 checkpoint 空,服务层从 DB 消息表回灌一次。

**D8 可观测:每轮两行。**
`model_ctx`(主力 Agent 轮:摘要全文 + 滑窗逐条 + 条数 + token 估算)、`history_ctx`(指代/意图共用的历史:摘要行 + 滑窗,每轮必打,闲聊兜底轮也打)。

## 不做什么

- 语义检索捞历史、主题重要度留关键事实(需求「本章不做」)
- 跨会话记忆、用户画像
- 摘要的合并重写(旧的只作背景)

## 验收映射

| 验收 | 怎么验 |
|---|---|
| ① 20 轮不爆不崩 | 脚本连发 20 轮,断言不出现 error 事件、每轮 `model_ctx` 的 tokens 不超 window |
| ② 演示配置走完级联 + 靠梗概答对早期订单 | `MODEL_CONTEXT_WINDOW=18000` 等五项,日志里出现 `layer1 降级`、`summary trigger`、`summary done 第N段`;再问「最开始那个订单后来怎么说」 |
| ③ 默认窗口 20 轮不降级不摘要 | 同 ①,断言日志无 `summary` / `layer1 降级` |
| ④ 摘要不阻塞 + grep `model_ctx`/`history_ctx` | 记摘要耗时与当轮回复耗时,前者在后者之后完成;日志可 grep |
| ⑤ 侧栏多会话 | `GET /api/conversations`、`GET /api/conversations/{id}/messages` + 前端侧栏 |
