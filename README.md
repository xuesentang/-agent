# 电商智能客服系统

电商客服 Agent 系列：CH01 纯对话 + CH02 Function Calling 工具链 + CH03 RAG 知识库 + CH04 混合检索与重排（含引用溯源、拒答、评估体系）+ CH05 工作流编排与主力 Agent + CH06 分流器正式版（指代消解、意图八分类、退款子流程、槽位回填）+ CH07 会话上下文管理（三层结构、异步摘要、预算倒推、多会话侧栏）+ CH08 工具系统（注册中心、执行引擎、审计、MCP 接入、写操作确认流）。FastAPI + SQLAlchemy(Async) + LangChain + **LangGraph**，模型接入走 OpenAI 协议（当前配置为 DeepSeek-V4.1-Flash，换供应商只改 `.env`）。

## 功能

| 接口 | 说明 |
|---|---|
| `POST /api/v1/chat/stream` | 多轮对话，SSE 流式。CH05 起走 LangGraph：指代消解 → 意图识别 → 五路分流 →（知识/退款类）检索 + 置信度闸 → 主力 Agent（ReAct，可多步调工具）。事件含 `node`（节点进出）、`resolved`（补全后的问题）、`intent`、`gate`、`tool_start`/`tool_end`、`order_options`（订单卡片）、`citations`、`done`（带 `actions`）。CH06 起请求可带 `order_id`（前端点选订单卡片后回填） |
| `POST /api/v1/chat/actions/transfer` | 转人工（前端模拟）：返回「已转接人工客服」+ 客服小猫问候，只留痕不建单 |
| `POST /api/v1/chat/actions/create-ticket` | 建工单：走 CH02 的 `create_ticket` 工具写 `tickets` 表，与转人工互不绑定 |
| `GET /api/v1/refund/reasons` | 退款原因固定类目（8 个），退款单下拉用 |
| `POST /api/v1/refund/apply` | 提交退款单：受理即建一张「售后」工单，工单号当受理号返回 |
| `POST /api/v1/chat/resume` | CH08：工单预览卡片上点「确认提交 / 取消」后接着跑被挂住的那一轮（`{conversation_id, decision, ticket_type, description}`） |
| `GET /api/v1/conversations` | 只读：该用户全部会话（新的在前、首问预览、是否已摘要），侧栏用 |
| `GET /api/v1/conversations/{id}/messages` | 只读：回载某会话的历史原文（不是梗概），切回旧会话用 |
| `POST /api/v1/chat/extract` | 售后描述结构化抽取：`order_id` / `claim_type`（八枚举）/ `expected_solution`，未提及为 `null` |
| `GET /` | 聊天页（原生 HTML/JS）：消息流、流式渲染、节点流程行、可点选订单卡片、退款单表单、工具轨迹徽章、可点引用角标（点开看来源章节与原文）、每条回答的 👍/👎 反馈 |

**RAG 进阶**（CH04）：

- **混合检索**：关键词路落在 Milvus 原生 BM25 上（`text` 字段挂 BM25 函数，中文用内置 jieba 分词），向量路 1024 维 COSINE；两路各召回 Top-50，用 `hybrid_search` 配 RRF（k=60）融合。
- **重排**：`bge-reranker-v2-m3` 对粗排 Top-20 精排出 Top-10；喂进 prompt 时按「最相关的放首尾」排列，避开中段注意力低谷。
- **元数据过滤**：`category` 作为 Milvus 标量字段，先过滤再检索。
- **Query 理解**：口语问法先归一回标准问法（LLM，失败回落原句），同义词只在检索侧扩展（「邮费」→「运费」），入库侧一字不改。
- **引用溯源**：证据在 prompt 里带 `[n]` 编号，编号与下发给前端的 citation 列表严格同源；答案里的角标点开即显示来源章节路径与原文。
- **拒答与低置信度池**：检索零命中、或生成前证据自评判「不够答」→ 明确拒答，并把原话落进 `low_confidence_questions`（来源/原因/会话）。闸门在生成之前——答复是流式的，吐出去再判就只剩事后道歉。
- **评估体系**：357 块（82538 字 / 15 篇）知识库，Top-10 只占 2.8%——这个数字决定了指标有没有意义。76 条用例 = 45 干净 + 31 噪声变体，四桶（政策直问 / 点名型号 / 只许黑话 / 一句问两件事）+ D 桶负例 14 条（far 6 / near 8，逐条核过「库里确实没有」）。**出题与标注分离**：模型只看目录出题，`expected_section` 与要点按事实骨架回填，人为改标全部记在 `scripts/assemble_eval_set.py` 的两张显式表里。检索段出 R@5/@10/MRR 四策略对比（算术平均 + 加权平均两套口径），生成段用 LLM 裁 Faithfulness 与相关要点覆盖，判编造的写进 `faith_cases` 台账。

**工作流编排**（CH05，`app/agent/`）：

- **图骨架**：指代消解 → 意图识别 → 按意图分流 →（知识类）知识检索 → 置信度闸 → 主力 Agent → 日志记录。能确定的事写在图里，只有不确定的（判意图、选工具、组织语言）交给模型。
- **七类意图 → 四个出口**：物流/订单/售后 → 业务数据类（不预检索，直进 Agent）；商品咨询/退款退货 → 知识类（强制检索 → 过闸 → Agent）；投诉 → 不进 Agent，安抚话术 + 「转人工」「建工单」两个选项；闲聊 → 固定话术、零模型调用。分流表在 `app/agent/intent.py`，改规矩改表不改 prompt。
- **置信度闸**（`app/agent/gate.py`）：卡在检索之后、进 Agent 之前——答复是流式的，说完再判只剩道歉。两层：top1 相似度 `< KB_MIN_SCORE`（0.45）直接判弱；过了阈值才花一次调用做证据自评。自评异常按「不够」处理（CH04 时是 fail-open，有了转人工出口后改 fail-closed）。
- **主力 Agent**（`app/agent/react.py`）：手写 ReAct 循环，三个停止条件（不再要工具 / 步数上限 `AGENT_MAX_STEPS=4` / token 预算 `AGENT_TOKEN_BUDGET=6000`）。缺订单号这类必填信息时由工具的参数校验回灌错误，模型据此追问。
- **祛魅对照**：`app/agent/naive_loop.py` 是同一件事去掉框架的写法（一个 while + 三个出口），`python -m app.agent.naive_loop "订单 1001 的物流到哪了"` 可以直接看它怎么转。
- **会话状态**：LangGraph State 贯穿全图，checkpointer 按 `thread_id = conversation_id` 存图状态；对话历史仍以 MySQL `messages` 表为权威源，每轮读出、裁剪、注入 State。

**工具系统**（CH08，`app/core/tools/`）：

- **注册中心一张表**：内置工具和 MCP 拉来的工具登记在同一份 `ToolRegistry` 里，每个挂 `ToolMeta`（来源 / 是否写操作 / 哪个 Server）。元数据当 sidecar，不动 `BaseTool` 协议——MCP 的 `StructuredTool` 不用包一层。重名直接抛，不做静默覆盖；MCP 接管 `query_logistics` 时在装配阶段就让内置那条不注册。
- **执行引擎**（`executor.py`）：存在性 → 权限 → 参数校验 → 执行，顺序不能换（不该执行的操作连参数对不对都不用告诉模型）。校验失败不抛异常，把说明包成工具结果回灌，模型据此追问或改参。超时统一在这里卡；只重试网络抖动类错误，**写操作永不重试**（超时未必没执行）。结果统一格式化并截断到预算内。
- **审计**（`tool_audit_logs`）：每次调用一条，含被权限拒、被校验拦的。表不挂外键，写审计失败只记日志——审计是旁路，不能反拦工具执行。
- **MCP 接入**：两个自建 Server（物流 8121 / 售后 8122，Streamable HTTP）。清单按 TTL 拉一次缓存，失败保留上次清单——Server 挂了不该让整个客服瘫。Server 侧加工具只重启那一个 Server，客服服务不重启就能用上。
- **写操作确认流**：`create_ticket` 走到执行引擎前先在图里挂住（`interrupt`），把类型 + 描述推成前端预览卡片；用户点「确认提交」→ `POST /chat/resume` → `Command(resume=...)` 接着跑。**没确认的写操作一律拒**，不给模型绕过去的机会；恢复后落库用的是用户卡片上看到的那份参数，不是模型重跑后的版本。

**上下文管理**（CH07，`app/core/context.py` + `app/core/token_budget.py` + `app/services/summary_service.py`）：

- **三层结构**：最近的留原文（层 1）、中间按角色截短（层 2：用户原话不动、客服答复只留开头 40 字、工具结果换成一行标识）、最远的压成梗概（层 3）。切分靠 `conversations` 上两个锚点 `summary_upto_msg_id` / `layer1_from_msg_id`，**降级只挪 id、消息行永远不动**——否则切回旧会话就看不到原文了。
- **预算从窗口倒推**：`history = window - system(实测) - 工具定义 - 证据 - 摘要 - 输出预留 - 用户输入预留 - ReAct 峰值 - 5% 安全余量`，再按 7:3 分层 1 / 层 2。参数全在 `.env`，不写死。
- **token 口径自己数**：LangChain 的 `count_tokens_approximately` 按英文 BPE 数中文，1061 个汉字报 5305（五倍）。改用汉字 1 token、其余 4 字符 1 token，对中文偏保守。
- **异步摘要**：层 2 的**用量**（截断前）超预算才在后台起一次压缩，压成一段追加进 `conversation_summaries`，边界追到层 1 起点；不 await、失败只记日志。触发看用量不看条数——工具结果不落消息表，数条数看不见它涨。
- **拼装顺序**：`system`（人设 + 红线 + 工具定义）→ 层 2 → 层 1 → 当前这句 →（梗概 + 检索证据）。梗概不单独占 system：上游会把所有 system 上提合并，工具定义被挤到可变内容之后，前缀缓存整段作废。
- **可观测**：每轮 `history_ctx`（摘要行 + 滑窗，闲聊兜底轮也打）、主力 Agent 轮 `model_ctx`（梗概全文 + 滑窗逐条 + 条数 + token），后台摘要的 trigger / start / done / skip / fail 全留痕。
- **多会话**：State 的 `messages` 挂 `add_messages` reducer 存全量（checkpoint 落盘），喂模型的是另一份裁过、带梗概的精简版，两者互不影响；前端侧栏列出全部会话，点开回载原文接着聊。

**分流器正式版**（CH06，`app/agent/`）：

- **指代消解 + 问法归一并成一次调用**（`query_understanding.py`）：读同一份上下文的两个活分两次调，会得出两种「用户在说什么」；合并后一份 JSON 出 `resolved` / `canonical` / `order_id`。问题本身完整就 `changed=false` 原样透传——改写不是目的，把好问题揉一遍检索更差。
- **意图八分类**（`intent.py`）：七类业务意图 + 「其他」兜底，prompt 按四件套写（枚举成选择题 / 强制 JSON 只出 `intent` 与 `confidence` / 边界 few-shot / 拿不准归「其他」）。判意图交给模型、去哪交给 `INTENT_ROUTES` 表，改规矩改表不改 prompt。
- **五个出口**：物流/订单/「其他」→ 业务数据类直进 Agent；商品咨询 → 知识类（强制检索 + 过闸）；**退款退货/售后 → 退款子流程**；投诉 → 安抚 + 两个独立按钮；闲聊 → 固定话术零模型调用。
- **退款子流程**（`refund_flow.py`）：顺序写死，不由模型决定——订单号 → `query_order` 拿这一单的事实 → 请求扩写（≤3 条，`chunk_id` 去重取最高分）检索政策 → 过闸 → 只把「这一单能不能退」交给主力 Agent 判断。退款判错一次的代价是赔钱加纠纷，而需要的证据永远是固定两样。
- **槽位不回填就不猜**：缺订单号时子流程停住，下发 `order_options`（该用户近期 3 单）渲染成聊天流里的可点选卡片；点选后用**同一句原话 + 订单号**重发，子流程从断点继续。退款原因不追问，提交退款单时从固定类目下拉选。
- **扩写只给容错低的场景**：退款退货/售后才多路扩写，简单 FAQ 不扩；知识库只存一份，扩写在检索侧现查现用。

**五个工具**（`app/core/tools/business_tools.py`）：`query_order` / `query_product` / `query_logistics`（工具内 mock 随机生成，同入参结果确定）、`query_faq`（CH03 起内部为向量语义检索：BGE-M3 嵌入 → Milvus Top-5 → 回 MySQL 取原文；向量库故障时降级 SQL LIKE 兜底）、`create_ticket`（写 tickets 表）。CH08 起这五个都挂进注册中心（带来源 / 读写元数据）；`MCP_ENABLED=true` 时 `query_logistics` 让位给物流 MCP Server，另有五个 MCP 只读工具（物流轨迹 / 送达预估 / 是否在保 / 退货进度 / 发票状态）。

**RAG 知识库**（`app/rag/`，CH03）：

- **离线建库**：Markdown 知识文档按标题层级结构感知切分（超长递归切、块间重叠裁到最近句号、大表格按行切且每块复制表头）；`category + questions + answer` 三格拼成向量化文本，章节路径 / 内容类型 / 关键条款标记 / 前后块指针四类元数据只存不进向量。
- **双写落库**：MySQL `knowledge_chunks` 是原文权威源，Milvus `knowledge` 集合存向量（主键与 chunk id 严格对齐）。先写 MySQL 记 `pending`，嵌入后 upsert Milvus 回填 `vector_id` 转 `done`；按主键幂等，中断后重跑只补 pending。
- **对话挖知识**：`app/rag/miner.py` 把历史客服对话分批喂 LLM 抽 QA 对，先进 `qa_extraction_staging` 暂存、整体去重（规范化问法）后入知识库；服务端定时开关 `MINE_QA_ENABLED`（默认关）。
- **在线检索**：问题向量化 → Milvus 相似度 Top-K → 回 MySQL 取原文，替换 `query_faq` 的关键词实现（工具入参出参契约不变）。

**多轮上下文**：历史落 MySQL `messages` 表（权威源），state 里另存一份全量喂 checkpoint。CH07 起不再用 `HISTORY_TOKEN_BUDGET` 单值裁剪，改按窗口倒推的三层结构（见上）；`HISTORY_TOKEN_BUDGET` 仍保留给未接入上下文管理的脚本用。同一 `session_id` 连续发消息复用最近进行中的会话，未知 `conversation_id` 静默走复用/新建。

## 快速开始

```bash
# 1. 装依赖（Python 3.13；torch 用 CPU 源）
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 配置
cp .env.example .env   # 填入 LLM_API_KEY（智谱开放平台密钥）

# 3. 起数据库（Docker MySQL 8.4，主机端口 3307）
docker compose up -d

# 4. 建表 + 灌 FAQ 种子数据（AI 合成，注明见 scripts/seed_faq.py）
python scripts/apply_schema.py
python scripts/seed_faq.py

# 5. 下载 BGE-M3 权重到 models/bge-m3/（~2.3G，HF 镜像直下；
#    HF_ENDPOINT 走 hf-mirror 时 Xet 后端会 401，用 resolve 直链 curl）
mkdir -p models/bge-m3/1_Pooling
for f in config.json config_sentence_transformers.json modules.json \
         sentence_bert_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json sentencepiece.bpe.model; do
  curl -L -o "models/bge-m3/$f" "https://hf-mirror.com/BAAI/bge-m3/resolve/main/$f"
done
curl -L -o models/bge-m3/1_Pooling/config.json \
  "https://hf-mirror.com/BAAI/bge-m3/resolve/main/1_Pooling/config.json"
curl -L -o models/bge-m3/pytorch_model.bin \
  "https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin"

# 6. 建知识库（种子文档 → 切分 → 双写 MySQL+Milvus）
python scripts/seed_knowledge.py      # 生成 data/knowledge_docs/ 三份 AI 合成文档
python scripts/build_knowledge.py     # 默认 Milvus Lite 文件库 ./data/milvus_lite.db

# 7. 起服务（lifespan 会预热嵌入模型，首条查询不背加载耗时）
python -m uvicorn app.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

向量库存放位置由 `VECTOR_STORE_URI` 决定：默认 `./data/milvus_lite.db`（Milvus Lite 嵌入式文件库，零 Docker 依赖）；切 Milvus 服务端只改这一行（如 `http://127.0.0.1:19530`）。docker-compose.yml 里留有 Milvus standalone 三件套注释，需要时取消注释。

## 验收命令（CH03 两条标准）

```bash
# ① 换说法问题能召回并答对：「邮费是多少」全库只有「运费」字样，LIKE 必漏召回，
#    向量语义检索应命中运费规则
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-3", "message": "邮费是多少"}'
# 期望: tool_start(query_faq) → tool_end(ok=True, 命中运费规则) → token… → done

# ② 中断重跑不丢块：跑一批后中断 → 重跑 → pending 补齐、Milvus 无重复向量
python scripts/build_knowledge.py --batch-size 8 --stop-after 1
python scripts/build_knowledge.py --batch-size 8
# 期望: 第一次 interrupted(剩 pending)，第二次 done 且 vectorized 补齐剩余
```

## 验收命令（CH05 五条标准）

```bash
# 起服务（本机开发库 + Milvus Lite 开发库，避开 .env 里的 MySQL 配置）
DATABASE_URL="sqlite+aiosqlite:///./data/devstack.db" \
VECTOR_STORE_URI="./data/devstack_milvus.db" \
python -m uvicorn app.main:app --port 8010

# ① 政策类问题：强制检索节点被走到
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" -d '{"session_id":"ch05-a","message":"邮费是多少"}'
# 期望: event: intent(route=knowledge) → node(retrieve_knowledge) → node(confidence_gate)
#       → gate(passed=true) → node(agent) → token… → citations → done
# 日志: grep "agent.trace" 能看到 intent/route/gate_passed/top_score

# ② 「订单 1001 的物流到哪了」：Agent 自己调工具
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" -d '{"session_id":"ch05-b","message":"订单 1001 的物流到哪了"}'
# 期望: route=business（不预检索）→ tool_start(query_order) → tool_start(query_logistics)

# ③ 我要投诉：两个独立按钮，分别点、互不绑定
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" -d '{"session_id":"ch05-c","message":"我要投诉"}'
# 期望: done 事件 actions=[transfer_human, create_ticket]
curl -s --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/actions/transfer \
  -H "Content-Type: application/json" -d '{"conversation_id":<上一步的 id>}'
curl -s --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/actions/create-ticket \
  -H "Content-Type: application/json" -d '{"conversation_id":<同上>}'
# 期望: 转人工只回问候、不建单；建工单回 ticket_no，tickets 表新增一行

# ④ 闲聊：固定话术
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" -d '{"session_id":"ch05-d","message":"你好呀"}'
# 期望: route=chitchat, finish_reason=chitchat, 无 tool_start、无 citations

# ⑤ 多步 ReAct
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ch05-e","message":"我订单 1001 现在什么状态，帮我确认下这单的物流走到了哪一步"}'
# 期望: 两次 tool_start（query_order → query_logistics）
```

## 验收命令（CH06 四条标准）

```bash
# 起服务同上（devstack 库，端口 8010）

# ①② 多轮意图与指代：6 个多轮场景 + 6 条怪问题/边界题，走真实模型
python -m tests.eval.run_dialogue_eval --out reports/ch06_dialogue.md
# 期望: 意图 16/16、指代补全 8/8、怪问题落「其他」5/5
python -m tests.eval.run_intent_eval --out reports/ch05_intent.md
# 期望: 38/38，八分类 prompt 不退化

# ③ 指代消解看得见：同一会话两轮，第二轮说「这个能退吗」
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ch06-a","message":"订单 1003 的物流到哪了","new_conversation":true}'
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" -d '{"session_id":"ch06-a","message":"这个能退吗"}'
# 期望: event: resolved(resolved="订单 1003 的包裹能退吗", changed=true, order_id="1003")
#       → intent(route=refund) → node(refund_subflow) → gate(passed=true) → token(订单事实+政策条款) → done

# ④ 缺订单号先弹选择器，点选后子流程接着走
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ch06-b","message":"我要退货","new_conversation":true}'
# 期望: event: order_options(3 笔) + done.finish_reason="need_order"
curl -sN --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"ch06-b","message":"我要退货","order_id":"<上一步某笔订单号>"}'
# 期望: gate(passed=true) → token(先报订单事实再追问质量/无理由) → done.actions=[apply_refund]

# 退款单提交（前端表单点「提交申请」走同一接口）
curl -s --noproxy '*' -X POST http://127.0.0.1:8010/api/v1/refund/apply \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":<上一步 id>,"order_id":"1018","reason":"质量问题"}'
# 期望: {"status":"submitted","refund_no":"T…"}，tickets 表新增一行「售后」
```

**前端手测**（浏览器打开 `http://127.0.0.1:8010/`）：新会话发「我要退货」→ 聊天流里出现三张订单卡片 → 点一张，卡片锁成选中态并带上订单号重发 → 流程行走到 `refund_subflow`、答复里先报订单事实 → 点「申请退款」弹出表单、原因下拉 8 个类目 → 提交后系统气泡给出受理单号。CH04 遗留的两项一起看：答复里的 `[n]` 角标点开有原文、👍/👎 点一次即锁定。

## 验收命令（CH07 五条标准）

验收脚本自己起库、自己装配依赖（真模型 + 真向量库），不用起服务——要看的是「每轮实际发出去什么」，日志和轮次一一对应才好 grep。**跑之前先确认没有别的进程占着向量库**（Milvus Lite 是文件锁，8010 的服务开着就会 `PermissionError: LOCK`，检索全废但日志只有一行 WARNING）。

```bash
# ①②③ 默认窗口 20 轮：不爆、不崩，且一次都不该压（装得下就不压）
DATABASE_URL="sqlite+aiosqlite:///./data/ch07.db" VECTOR_STORE_URI="./data/ch07_milvus.db" \
  python scripts/context_demo.py --turns 20 --expect-cascade off
# 期望: failures 0、demotions 0、summary_triggers 0、model_ctx/history_ctx 每轮各一次、anchors (0, 0)

# ②④ 演示配置 90 轮：窗口压到 18000，看完整级联（层1 降级 → 层2 攒满 → 摘要落库）
#     45 轮不够：预算按真实用量累加，45 轮时层 2 才攒到 1000 token（预算 1693）
DATABASE_URL="sqlite+aiosqlite:///./data/ch07.db" VECTOR_STORE_URI="./data/ch07_milvus.db" \
  python scripts/context_demo.py --turns 90 --demo-config --expect-cascade on
# 实测: demotions 49 → summary trigger 层2 约 1700 > 预算 1693 → done 第1段 311 token(覆盖 msg 419~511)
#       → 再攒 1731 → done 第2段；failures 0；anchors (513, 559)

# ② 靠梗概答对早期订单：级联之后问「最开始那个订单后来怎么说」
#    （脚本第 16 题就是这句，答案里要出现早期报过的订单号 1001/1003 与对应诉求）
grep -A2 "最开始那个订单" <日志文件>

# ④ 摘要不阻塞 + 两个 ctx 必打：直接 grep 日志
grep -c "model_ctx" <日志文件>       # 进 Agent 的轮次
grep -c "history_ctx" <日志文件>     # 每轮一次，闲聊兜底轮也有
grep "summary \(trigger\|start\|done\|skip\|fail\)" <日志文件>

# ⑤ 侧栏多会话切换回载：浏览器开两个会话各聊两句 → 左侧点回第一个
#    期望: 历史原文逐条回载、接着问「那它能退吗」指代仍能补全
curl -s --noproxy '*' 'http://127.0.0.1:8010/api/v1/conversations?user_key=ch07-a&limit=5'
curl -s --noproxy '*' 'http://127.0.0.1:8010/api/v1/conversations/<id>/messages?user_key=ch07-a'
# 期望: 前者 {"conversations":[{conversation_id, status, preview, message_count, summarized, created_at}]}
#       后者 {conversation_id, summarized, messages:[{role, content, tool_call_id, has_tool_calls}]}
#       user_key 就是聊天接口的 session_id（本项目没有用户体系，见已知限制）
```

**为什么演示配置能算出 18000**：需求给的那组参数是 `MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`，倒推得滑窗 5640（层 1 3947 / 层 2 1693）。实测与需求那组（5650 / 3954 / 1695）差 9~10 token，差在 system prompt 与工具定义的实测字数，算式一致——测试里用近似断言而不是把常量反标定过去。


## 验收命令（CH08 七条标准）

本章有两条需要 MCP Server 在跑。先起两个自建 Server（各一个进程，都是 Streamable HTTP）：

```bash
python scripts/mcp_logistics_server.py --port 8121   # query_logistics_track / estimate_delivery
python scripts/mcp_aftersale_server.py --port 8122    # check_warranty / query_return_progress / query_invoice_status
```

```bash
# ① 新工具只注册、不改核心代码
python scripts/tool_demo.py register
# 期望: 运行期注册 query_coupon 并调用成功;再注册一次同名被拦("工具重名")

# ② 参数校验回灌模型（不抛异常）
python -m pytest tests/test_tool_permission.py -q
# 期望: 内置工具走 pydantic 全量校验、MCP 工具(JSON Schema dict)只查必填项;拦下记「校验拦下」

# ③ 内置 + MCP 同一份表；Server 加工具只重启该 Server
MCP_ENABLED=true MCP_SERVERS="logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp" \
  python scripts/tool_demo.py registry
# 期望: 9 个工具一张表,来源列 builtin/mcp、写操作只有 create_ticket、Server 列能对上
# 加工具的做法: 在 mcp_aftersale_server.py 里加 @mcp.tool(),只重启 8122,重跑本命令即见
# （内置 query_logistics 在 MCP_ENABLED=true 时不在清单里——被接管就下线,不留重名）

# ④ MCP 接管 + 真调一次 + 审计带来源
MCP_ENABLED=true MCP_SERVERS="..." python scripts/tool_demo.py mcp
# 期望: 内置 query_logistics 不在清单;query_logistics_track ok;审计 tool_source=mcp / mcp_server=logistics

# ⑤⑥ 拒绝不落库 / 超时与重试规矩
python scripts/tool_demo.py confirm --decision reject
# 期望: 先出卡片、此时 tickets 0 行 → 取消后仍 0 行、审计「权限拒绝」、答复改口「没有提交工单」
python scripts/tool_demo.py resilience
# 期望: 只读抖动重试 1 次成功;只读超时 retry=0;写操作超时 retry=0 且只调 1 次

# ⑦ 确认后接着跑完
python scripts/tool_demo.py confirm --decision approve
# 期望: 卡片 → 确认 → create_ticket ok → tickets 1 行(T{日期}{序号}) → 消息流水 user/assistant/tool/assistant

# 想看真模型自己会不会建单(不稳,只作抽查)
python scripts/tool_demo.py confirm --real-model
```

**前端手测**（`http://127.0.0.1:8010/`）：问一句会触发建单的话（如「订单 1001 物流显示已签收但我没收到,帮我建个工单」）→ 聊天流里出现黄底预览卡片（类型 + 描述 + 确认提交/取消）→ 点「确认提交」，卡片变绿并附「已确认,正在提交…」，同一气泡继续流式写答复；点「取消」则变灰、答复改成「没有提交工单」。

`tool_demo.py` 默认落 CH07 那套开发库（`./data/ch07.db` + `./data/ch07_milvus.db`），需要覆盖就传 `DATABASE_URL` / `VECTOR_STORE_URI`。

## 验收命令（CH04 四条标准）

```bash
# ① 四策略对比报告（Recall@5/@10、MRR 分桶 + Faithfulness + 拒绝率）
python -m tests.eval.run_rag_eval --no-generation      # 只算检索指标（快）
python -m tests.eval.run_rag_eval --out reports/ch04_eval.md

# ② 带具体实体的问法 BM25 那路能命中:B_model 桶(宽楦/断底/开线/税号/到货提醒/防伪)
python -m tests.eval.run_rag_eval --no-generation | grep B_model

# ③ 引用角标点开看原文:浏览器问「邮费是多少」
#    期望 SSE: tool_start(query_faq) → tool_end → token…(带 [1]) → citations → done
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-4", "message": "邮费是多少"}'

# ④ 知识库没有的内容明确拒答并进池
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-5", "message": "你们支持货到付款吗"}'
# 期望: done 事件带 "finish_reason": "abstain" 与 "abstain_source"
# 池子: SELECT source, raw_question, reason FROM low_confidence_questions ORDER BY id DESC LIMIT 5;
```

MySQL 真库复跑（若在 SQLite/隔离库上做过验证）：`python scripts/build_knowledge.py --reset-store --batch-size 8 --stop-after 1` 后不带 `--stop-after` 重跑即可，`--reset-store` 会删掉旧集合重建（FLAT 索引）。

## 测试与评测

```bash
python -m pytest                            # 418 项单测（SQLite 内存库覆盖数据层）
python -m tests.eval.run_eval.py            # CH01 抽取 eval，13 条（需真实 key）
python -m tests.eval.run_tool_eval.py       # CH02 工具选择 eval，13 条（需真实 key）
python -m tests.eval.run_recall_eval.py     # CH03 换说法召回 eval，14 条（需真模型+真库）
python -m tests.eval.run_rag_eval           # CH04 四策略检索 + Faithfulness + 拒答（需真模型+真库）
python -m tests.eval.run_intent_eval        # CH05 意图识别，38 条标注样例（需真实 key）
python -m tests.eval.run_dialogue_eval      # CH06 多轮指代 + 意图 + 怪问题兜底（需真实 key）
python scripts/context_demo.py --turns 20 --expect-cascade off   # CH07 上下文：装得下就不压（需真模型+真库）
python scripts/tool_demo.py register        # CH08 工具系统：注册即上线 / MCP 接管 / 韧性 / 确认流
```

CH04 的评估集是**生成出来的**，脚本按顺序跑（`--reset-store` 重建前会先把 MySQL 的 `vectorize_status` 退回 pending，否则旧向量状态会让块被永久跳过）：

```bash
python scripts/expand_knowledge.py          # 骨架 → 12 篇核心文档正文（事实逐字回检）
python scripts/build_knowledge.py --docs data/knowledge_docs --reset-store
python scripts/gen_eval_cases.py            # 只看目录出题（出题与标注分离）
python scripts/assemble_eval_set.py         # 人工改标 + 规则噪声 + 合并 D 桶 → ch04_cases.jsonl
python -m tests.eval.validate_rag_cases     # 自检：标注对不上库就在这里挡掉
python scripts/spotcheck_eval.py --auto 12  # 人工抽检材料（完整答案 + 证据原文）
```

eval 阈值 90%。最近真实运行：CH01 抽取 12/13 = 92%；CH02 工具选择 **13/13 = 100%**（连跑两轮稳定）；CH03 换说法召回 **14/14 = 100%**（真 BGE-M3 + Milvus，Top-5 命中判定）；CH05 意图识别 **38/38 = 100%**；CH06 多轮意图 **16/16**、指代补全 **8/8**、怪问题落「其他」**5/5**（`reports/ch06_dialogue.md`）。

CH04 四策略 × 分桶（**357 块库 / 31 道干净题**，真 BGE-M3 + 真重排）：

| 策略 | R@5 | R@10 | MRR |
|---|---|---|---|
| dense | 0.81 | 0.94 | **0.709** |
| bm25 | 0.48 | 0.81 | 0.394 |
| hybrid | **0.84** | 0.94 | 0.554 |
| hybrid_rerank | 0.81 | 0.90 | 0.664 |

结论不是一个数：**hybrid 的 R@5 高于 dense（多路召回有用），但 MRR 低 0.155（RRF 把正确答案从第 1 位挤到第 2-3 位）**。重排把 MRR 救回 0.664（A 桶 0.417→0.736），救不回 dense 的水平，还会把 R@10 压到 0.90——候选池外的块重排够不着。默认留 `hybrid_rerank`，要首位准确就该用 dense，两者不能同时最优。

生成段 Faithfulness 31/31、相关要点覆盖 51/58；拒答 far 6/6、near 8/8。**这三个数都别当成绩看**：裁判是生成模型自己（`JUDGE_MODEL` 空着会回落，报告里标红）、near 全拒说明题太容易、人工抽检 12 条抓出 1 条「检索没捞到却答成知识缺失」。报告见 `reports/ch04_eval_v2.md`，抽检材料见 `reports/ch04_spotcheck.md`，扩库前那份 21 块基线留在 `reports/ch04_retrieval_baseline.md` 作对照。

## 关键配置（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容上游（DeepSeek 官方；换别家只改这三行） |
| `LLM_MODEL` | `deepseek-flash` | 模型名（DeepSeek-V4.1-Flash；非思考，长文生成比思考模型省六到七成 token） |
| `LLM_API_KEY` | — | 密钥，必填 |
| `LLM_STRUCTURED_METHOD` | `json_mode` | 抽取的结构化输出方式（DeepSeek 的 JSON 模式要求提示词里出现 “json”，本项目提示词均已满足） |
| `LLM_THINKING` | `disabled` | DeepSeek 思考模式开关。开着时长输出 9398 token / 36 秒，关了 764 token / 6 秒；`tools` 场景还需回传上一轮 `reasoning_content`，本项目未做，故默认关 |
| `RECALL_ROUTE_LIMIT` | `20` | 混合检索路由候选池大小（扩库后从 50 降到 20，否则重排白算） |
| `JUDGE_MODEL` | 空 | 评估裁判模型。空则回落生成模型，报告里标红「裁判与生成同模型」，绝对分数不可对外 |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` | 空 | 跨厂商裁判时的独立凭据（建议和生成用不同厂商） |
| `DATABASE_URL` | `mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent` | 数据库连接（3307 避让本机 3306） |
| `MYSQL_ROOT_PASSWORD` | — | docker-compose 的 root 密码 |
| `DB_ECHO` | `false` | SQL 日志开关 |
| `TOOL_TIMEOUT_SECONDS` | `10.0` | 工具执行超时 |
| `HISTORY_TOKEN_BUDGET` | `2048` | 旧的口径单值裁剪（CH07 起由窗口倒推取代，仅遗留脚本用） |
| `VECTOR_STORE_URI` | `./data/milvus_lite.db` | 向量库地址（不能叫 `MILVUS_URI`，与 pymilvus 内部配置撞名） |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型（本地目录或 HF 名） |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型（本地目录或 HF 名，~2.3G） |
| `EVIDENCE_SELFCHECK_ENABLED` | `true` | 生成前的证据自评闸；关掉则只按「零命中」拒答 |
| `KB_MIN_SCORE` | `0.45` | 知识类置信度闸的分数粗筛线（dense 相似度；实测库内外重叠，只能挡明显离题的） |
| `AGENT_MAX_STEPS` | `4` | 主力 Agent 单轮 ReAct 的步数上限 |
| `AGENT_TOKEN_BUDGET` | `6000` | 主力 Agent 单轮 token 预算，见底即停并交出已查到的结果 |
| `MODEL_CONTEXT_WINDOW` | `65536` | 模型上下文窗口（CH07 预算倒推的起点；换供应商必须改这个数） |
| `MAX_OUTPUT_TOKENS` | `2000` | 单轮答复预留 |
| `MAX_USER_INPUT_TOKENS` | `2000` | 用户单句预留 |
| `TOOL_RESULT_MAX_TOKENS` | `1200` | 单条工具结果预留（决定 ReAct 峰值那笔） |
| `RERANK_TOP_K` | `5` | 检索证据条数（决定证据那笔预留） |
| `LAYER2_ASSISTANT_HEAD` | `40` | 层 2 里客服答复保留的开头字数 |
| `CONTEXT_SUMMARY_ENABLED` | `true` | 后台摘要总开关 |
| `CONTEXT_SUMMARY_TIMEOUT` | `60` | 单次摘要超时（秒），超时只记 `summary fail`，不影响主链路 |
| `LOG_LEVEL` | `INFO` | 日志级别；`agent.trace` 一行一轮，验收靠它 |
| `MINE_QA_ENABLED` | `false` | 对话挖知识定时任务开关 |
| `MINE_QA_INTERVAL_HOURS` | `24` | 挖知识定时间隔 |
| `MINE_QA_BATCH_SIZE` | `20` | 每批挖掘的会话数 |
| `MCP_ENABLED` | `false` | 打开后内置 `query_logistics` 下线，物流交给 MCP Server 接管 |
| `MCP_SERVERS` | 空 | `名字=URL` 逗号分隔，如 `logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp` |
| `MCP_TOOL_TTL_SECONDS` | `30` | 工具清单缓存时长：Server 侧加工具后最多等这么久，客服服务不用重启 |
| `MCP_TIMEOUT_SECONDS` | `20.0` | MCP 调用超时（走同一个执行引擎的超时） |

## 已知限制

1. ~~只做单轮工具调用~~ CH05 起主力 Agent 是多步 ReAct（上限由 `AGENT_MAX_STEPS` 控）。CH07 起会话内上下文有正经管理（三层 + 后台摘要 + 预算倒推），但仍**不做跨会话记忆与用户画像**——每通会话独立，`user_key` 只是个会话归属键。
2. CH04 起默认走 **hybrid_rerank**；dense 与 BM25 的分数不同量纲、RRF 分只反映名次，跨策略只能比 Recall/MRR，不能比分值。LIKE 降级路仍保留为向量库故障时的兜底。
3. 向量库存 Milvus Lite 文件库（单机嵌入式，无 Docker 依赖）：索引定死 FLAT 暴力精确检索——一是演示量级（几百块）暴力扫更准，二是 milvus-lite 的 AUTOINDEX 走 faiss，其 Windows 窄字符文件 API 写不了中文路径，跨进程重开必崩（实踩，见 dev-notes/ch03.md）；集合存主键+向量+text(供 BM25 建索引)+category(供过滤)，原文正文一律回 MySQL 取。
4. 挖知识的 QA 去重只做规范化精确/包含比对，语义去重仍待补；挖出的 QA 用固定分类标签「对话挖掘」，类目归并入 CH09 飞轮。
5. 引用编号只保证「模型用了返回里出现过的编号」，不保证编号与句子语义严格对应（模型可能把 [2] 标在 [1] 的内容后）；要强约束得做逐句归因，留给后续。
6. mock 三工具（订单/商品/物流）函数内随机生成，不接真实系统（时间基准为固定常量，同入参跨天调用逐字一致）；`create_ticket` 工单号当日序列生成，撞号有 `IntegrityError → seq+1` 重试兜底，但取号本身非原子，极端并发下仍可能连续撞号（CH08 权限与审计章前以注释标注）。
7. 无鉴权，仅限本地/内网开发用；会话随服务重启仍在（DB 持久化），但无用户体系。
8. 上游为 DeepSeek-V4.1-Flash（非思考）。此前用智谱 GLM-5.3-Flash 时实测：65 字回答消耗 656 token（思考 token 占九成以上），930 字正文耗 64.9 秒，且上游对并发长请求直接排队（并发 4/6 会几分钟零产出，不是 429）——批量生成长文时这是硬约束，记录在 dev-notes/ch04.md。
9. BM25 中文分词用 Milvus 内置 **jieba**（服务端的 `chinese`/lindera 分词器未编进 milvus-lite）；换 Milvus 服务端后建议改回 `chinese`，代码里只改 `ANALYZER` 一处。
10. SSE `data` 为 JSON，中文走 `\u` 转义（前端解析无影响；CH08 工具结果序列化时统一处理中文不转义）。
11. 拒答闸以「知识库」为口径：同一轮里既查单/查物流又查知识库时，只要 `query_faq` 零命中就整轮拒答（连同轮的工具结果一起不答），保守取向——宁可少答一次也不让模型凭常识补政策。反方向放行会让「政策类问题」失去唯一的硬约束。
12. **写操作确认流恢复时会多调一次模型**：LangGraph 的 `interrupt` 在 resume 时会把整个 agent 节点从头重跑（ReAct 循环跟着重跑），实测多 1 次模型调用。落库内容以用户卡片上看到的那份参数为准（重跑后模型改口不影响落库内容），所以工具本身仍要求幂等。
13. **MCP 工具只做必填项校验**：适配器带过来的是 Server 那侧的 JSON Schema dict，没有 pydantic 模型，所以只查 `required`；类型/枚举这类深校验由 Server 自己负责，我们这侧不假装验过。内置工具仍走 pydantic 全量校验。
14. **MCP 清单是快照**：Server 删掉工具后旧清单不会自动消失（调用会报连接错），要重启客服服务或强制重拉。静默删掉模型正在用的工具更糟，所以不做自动清理。MCP 工具一律按只读登记——对方的用途声明不可信，写操作必须在我们这侧另配 `ToolMeta(read_only=False)`。
12. ~~证据自评失败按「证据够」放行~~ CH05 起改为 **fail-closed**（判不了就当不够，走兜底 + 两个出口）：有转人工之后，多拒一次的代价只剩用户多点一下。`degraded` 标记仍保留，入池原因能分辨「判了弱」和「没判成」。
13. 图状态的 checkpointer 是 `InMemorySaver`（框架自带、零依赖）：进程重启后图中间态清空，但对话历史在 MySQL 里，用户接着聊无感。要跨进程持久化需要 `langgraph-checkpoint-sqlite`，未装（不为一个功能动运行环境）。
14. 意图识别八类（七类业务 + 「其他」兜底），解析失败落「其他」（业务数据类，直进 Agent）——失败代价最小的方向。置信度字段只记录、不参与路由：本章按「默认配够格的大模型」走，成本吃紧时的小模型试判 + 低置信度升级重判的降级路没做。
15. 「转人工」是纯前端模拟：后端只落一条消息留痕，没有真人系统、没有会话状态机。
16. 订单数据全部来自 `app/core/mock_orders.py` 这一处（AI 合成，20 笔固定的，`order_view()` 是查单工具与订单卡片的唯一同源）。「近期订单」按 `user_key` 取,而 `user_key` 现在是 `session_id`——换会话就看不到「自己的」订单了;真系统里应该按用户 ID。
17. 退款单受理即建一张「售后」工单，工单号当受理号：没有审批流、没有退款状态机、不回写订单状态。真要跑通退款得有支付侧对接，本章只到「受理」。
18. 指代消解失败（上游抖动/JSON 不可用）会原样透传，表现是「这个能退吗」没补上订单号 → 弹订单选择器让用户点一下。兜底方向是对的（不许猜单号），但用户会多一次交互；日志里有 `指代消解失败,原样透传` 可查。
19. checkpointer 按 `thread_id` 跨轮保留 State，所以服务层每轮用 `_TURN_RESET` 显式重置本轮字段（否则上一轮的 `order_options`/`finish_reason` 会漏进本轮 done）。加本轮字段要记得加进那张表。
20. 上下文预算按**实测用量累加**（不是按每轮上限预留），所以级联什么时候发生取决于说了多少字：演示配置（窗口 18000）下要聊到 90 轮才压出第一段梗概。好处是不浪费窗口，代价是「聊几轮才压」不可预期。
21. `MODEL_CONTEXT_WINDOW` 是配置写死的（默认 65536），不按模型名查表——换供应商必须手动改这个数，忘了就会低估或高估窗口。
22. 摘要一段一条、写完不回炉重写，也没有任何质量校验：梗概里写错一个事实，它会一直躺在库里被当背景喂回去。要纠错只能人工改表。
23. 层 2 里工具结果一律压成一行标识（`[工具结果 N 字,已省略]`），下一轮追问同一单会再查一次工具——省 token 换来一次多余调用。
24. 层 1 降级按预算的 8 成微调，边界会随轮次移动，prompt 前缀缓存的命中率因此受影响；想稳住缓存得改成「一次挪到位、两次降级之间边界不动」，没做。

设计文档：`docs/superpowers/specs/`（ch01/ch02/ch03/ch04/ch05 各一份）；实施计划：`docs/superpowers/plans/`；开发留痕：`dev-notes/ch01.md` ~ `dev-notes/ch06.md`。
