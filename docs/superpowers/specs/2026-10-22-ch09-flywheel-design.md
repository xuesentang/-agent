# CH09 设计：数据飞轮闭环（无 Langfuse 版）+ 可观测的本地替代

> 需求原文：`各章提示词.md` 551-628 行。6 条验收。
> 前置事实：CH09 在 2026-10-21 因内存不足被用户叫停（Langfuse 自部署 = Docker Desktop + 6 容器
> 约 3~4G，机器 16G 当时只剩 3.2G）。CH10 已先做完（顺序与需求相反，本章不做分类器）。

## 0. 本章范围（用户 2026-10-22 拍板：「把第九章不依赖 Langfuse 的做了吧」）

需求六条功能拆开看，只有前两条非 Langfuse 不可：

| 需求 | 依赖 Langfuse | 本章 |
|---|---|---|
| 1 接 Langfuse，看完整 trace 树 | 是 | **不做**（等内存） |
| 2 意图进 trace 元数据、按意图统计 token | 部分（统计不必须靠它） | **做本地版**：按意图落本地表 + 聚合接口 |
| 3 三入口统一入池 + 召回片段快照 + 正式置信度闸 + 👎 接后端 | 否 | 做 |
| 4 飞轮流水线：标准化 → 查重 → review_queue → 人工审核 → 写回知识库 | 否 | 做 |
| 5 自动化评估流水线：复用 ch04 指标、eval_runs 落表、看趋势 | 否 | 做 |
| 6 前端待审队列后台页（列表 + 通过/驳回 + 详情看原话与召回片段） | 否 | 做 |

验收 6 条里，**第 1 条（Langfuse 里点开一条请求看链路）不做**，2~6 全做。README 与
dev-notes 里把这条缺口写在明面上，不假装做完了。

## D1 入池统一走一个函数，入口与快照一起落

`app/services/flywheel.py::enqueue(...)` 是唯一入口，三个调用点都传同一套参数：

| 入口 | source | reason 家族 | 召回快照 |
|---|---|---|---|
| 检索置信度闸拦下 | `retrieval_low_conf` | `no_hits` / `score_low` / `confidence_low` | 有（当轮 citations） |
| 生成前自评判「不够答」 | `self_check` | `self_check` / `self_check_degraded` | 有 |
| 用户点 👎 | `user_feedback` | `user_feedback` | 有则存、没有为 NULL |

`source` 枚举沿用 ch04 表定义（`retrieval_low_conf` / `self_check` / `user_feedback`），
不新造值——复盘口径不能因为加了一章就变。

## D2 表结构照 `各章建表.md` 605-640，只多一张 cost 表

- 新建 `review_queue`（需求 4 的 DDL，字段一字不改）、`eval_runs`（需求 5 的 DDL）。
- `low_confidence_questions` 加两列：`retrieved_chunks` JSON、`matched_review_id`（FK → review_queue.id）。
- **额外**建 `intent_usage`（需求 2 的本地替代）：`conversation_id`、`intent`、`route`、
  `prompt_tokens`、`completion_tokens`、`token_source`（`estimated` / `api`）、`model`、`created_at`。
  需求没给这张表的 DDL，因为需求假设「token 花销查 Langfuse」；本地替代必须自己存。
- MySQL 走 ALTER（`scripts/sql/ch09_schema.sql`），SQLite 开发台走 `create_all` + 幂等 `ALTER` 兼容。

## D3 👎 的召回快照由前端回传，不在 messages 表加列

需求说「👎 这种事后落池的，从该会话当轮的检索结果里尽力回捞」。两条路：

1. 把每轮 citations 存进 `messages`（加列）→ 要改 ch04 的消息表，且历史消息回填不了；
2. 前端手上就有当轮 citations（`citations` 事件里拿的），点 👎 时一起 POST。

选 2：DDL 不用动，且「回捞」语义天然成立——用户只能对自己刚看到的那条回答点 👎。
代价写在这里：**旧消息的反馈拿不到快照**（页面刷新后前端手里没有 citations），落池时
`retrieved_chunks` 为 NULL，审核人只能看原话。后端对回传内容做校验：最多 5 条、
每条只取 `chunk_id/section_path/score/snippet`，`snippet` 截 120 字，防止前端塞大对象。

## D4 evidence_confidence 用三个信号合成，阈值拿 ch04 评估集校准

需求点名「精排 Top1 的相关性分、有效证据数量、Top1 和 Top2 的分差」：

```
confidence = 0.60 * top1
           + 0.25 * min(有效证据数 / 3, 1)
           + 0.15 * min(max(top1 - top2, 0) / 0.30, 1)
```

- `有效证据数` = score ≥ `KB_MIN_SCORE` 的引用条数（沿用 ch04 的老阈值当「够不够沾边」的定义）；
- 分差归一化到 0.30：实测重排分差多在 0.05~0.25，0.30 以上再大也没新信息；
- 权重不调参（不做拟合），三个信号按「谁更可信」排序：单条分数最可信，数量次之，分差最弱。

**阈值不拍脑袋**：`scripts/calibrate_gate.py` 拿 ch04 那份 `ch04_cases.jsonl` 跑
——31 条干净题当正例、D 桶 14 条（far 6 / near 8）当负例，对每条真实检索结果算
confidence，扫 0.30~0.95 出曲线，取「负例拦下率最高、正例误杀 ≤ 10%」的点写进
`EVIDENCE_MIN_CONFIDENCE`。曲线与选定值落 `reports/ch09_gate_calibration.md`。

闸的位置不动（检索后、进 Agent 前），顺序：**confidence 粗筛 → 证据自评**。
confidence 替代原来的 `top1 < KB_MIN_SCORE` 单值判断（那个判断的实测毛病写在
`app/agent/gate.py` 文档里：库外问题 0.62 比库内 0.49 还高）。

## D5 飞轮流水线：先规则查重、再模型查重

`scripts/run_flywheel.py` 一次处理一批未归并的池内问题（`matched_review_id IS NULL`）：

1. **标准化**：一次 LLM 调用，把原话改写成 FAQ 式问题 + 顺带生成示例答案
   （结构化输出 `{normalized_question, ai_suggested_answer}`）。失败的问题不吞掉，
   计入 `failed` 留在池里等下一轮。
2. **查重**：先做**规则**匹配（规范化后完全相等 / 互相包含，长度差 ≤ 40%），
   命中直接累加 `occurrence_count`；规则不命中再让模型判（把待审队列里 top-20
   标准化问题连同新问题一起给它，结构化输出 `{matched_id | null}`）。
   规则前置的理由：池子里同一句话重复入池的情况很多（CH10 实测 39 行只有 13 个不同问题），
   这类重复不该花模型调用。
3. **归并落点**：无论命中还是新建，都把 `review_queue.id` 写回
   `low_confidence_questions.matched_review_id`。归并是「指过去」，原话行不删
   ——审核页要能看到「这条缺口由哪些真实问法贡献」。

## D6 审核通过 = 走 ch03 的双写流程，不许绕过去

`approve(review_id, approved_answer)`：

1. 建一行 `KnowledgeChunk`：`category="飞轮补充"`、`questions=normalized_question`、
   `answer=approved_answer`、`section_path="飞轮补充/人工审核"`、`content_type="qa"`；
2. `knowledge_repo.add_chunks` 落 pending → `vectorize_pending`（从 `build_index` 里
   抽出来的函数）embed + upsert Milvus（或开发台的 Lite 库）→ 回填 `done`；
3. `review_queue.review_status="通过"`、`approved_answer` 落库。

**驳回**只改状态，不写知识库。同一个缺口二次审核时要能覆盖答案（更新 chunk 而不是
再插一行）——用 `(category, questions, answer)` 去重是最省事的做法，与建库时的去重口径一致。

## D7 评估流水线复用 ch04 指标，指标算法抽成公共模块

`tests/eval/run_rag_eval.py` 里的 recall/MRR/加权口径是本章要复用的东西，但它写在
测试脚本里。抽到 `app/eval/metrics.py`，评估脚本改成 import（行为与输出不变，
有现成报告可对比），新写的 `app/services/eval_service.py` 调它并把结果写 `eval_runs`。

- 定时：`EVAL_PIPELINE_ENABLED` + `EVAL_PIPELINE_INTERVAL_HOURS`，默认关（跑一轮要加载
  嵌入+重排模型，开发台不该默认占 4G）。
- 手动：`scripts/run_eval_round.py --strategy hybrid_rerank [--with-generation]`。
- **生成段（Faithfulness）默认不跑**：每轮 31 次 LLM 调用，定时任务里烧钱；检索段
  （R@5/R@10/MRR）是每轮必跑的部分。`eval_runs.metrics` 里有 `with_generation` 标记，
  趋势图上两段分开画，不混在一条线里。

## D8 后台页 `/admin/review`：审核 + 趋势，一个页面两栏

左栏待审队列（标准化问题、出现次数、示例答案摘要、入队时间），点一行开右侧详情：

- 归并进来的用户原话（含来源入口、入池时间）
- 当时的召回片段快照（原文 + 得分），审核人对着片段判断「知识库真缺」还是「有但没检到」
- 核准答案输入框（预填 `ai_suggested_answer`）+ 通过 / 驳回

页面下方一条评估趋势（最近 N 轮 R@5 / MRR）。与 `/admin/topics` 同样的静态目录约定、
同一套只读接口风格：读写分离（GET 列表/详情，POST 通过/驳回）。

## D9 cost 统计：估算打底，标清来源

`intent_usage` 每轮一行：`intent`、`route`、`prompt_tokens`、`completion_tokens`、
`token_source`、`model`。

- 现在拿不到真实 usage：LangGraph 的 custom 事件里只有文本，ReAct 内部的
  `usage_metadata` 没往外传。真要采得改 CH02 的核心链路（`react.py` 累积 usage 再塞回 state），
  本章不动它。
- 因此用 `token_budget.estimate_tokens` 估算：prompt 侧用上下文预算算出的
  `layer1/layer2/system` 之和，completion 侧估答复文本。`token_source="estimated"` 写死在行里。
- 聚合接口 `GET /api/v1/usage/by-intent` 返回各类意图的 prompt/completion 总量与占比。
  文档里写明：**估算值同比例放大，能比「哪类烧钱」，不能当账单**。

## D10 本章的测试策略

- 可单测的全部走单测：repo（幂等累加、匹配落点）、标准化与查重的规则分支、
  confidence 公式、校准脚本的曲线计算、审批写库（Fake embedder/store）、
  eval 指标抽取后的等价性、API 契约。
- 不可单测的两件事，用真实数据跑一遍留痕：
  1. **验收 3 一整圈**：问一个库里没有的问题 → 落池 → 飞轮跑到待审 → 审核通过 →
     同一个问题再问能答对。真库（开发台 SQLite + Milvus Lite）+ 真嵌入模型。
  2. **校准曲线**：真检索跑 ch04 评估集，出报告。

## D11 与 CH10 的接口

CH10 的分类器读 `low_confidence_questions`，本章新增的 `matched_review_id` 与
`retrieved_chunks` 都是旁路列，不改它的查询（`list_all` / `list_unclassified` 都用
`select(LowConfidenceQuestion)`，加列不影响）。分布页与审核页各占一个后台路径，
`/admin/topics` 保持不动。

## D12 不做的事（写清楚免得被追问）

- 不做 Langfuse 接入（内存）、不做 trace 树、不做按节点的耗时瀑布图。
- 不做真实 token 账单（估算）。
- 不做审核权限/操作人（无用户体系，与全项目一致）。
- 不做飞轮的自动触发（定时跑标准化入库；审核必须人点）。需求没要求，且自动写知识库
  的风险比收益大。
