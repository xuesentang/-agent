# 电商智能客服系统

电商客服 Agent 系列：CH01 纯对话 + CH02 Function Calling 工具链 + CH03 RAG 知识库（向量语义检索）+ CH04 混合检索与重排（含引用溯源、拒答、评估体系）。FastAPI + SQLAlchemy(Async) + LangChain，模型接入走 OpenAI 协议（当前配置为智谱 GLM-5.3-Flash，换供应商只改 `.env`）。

## 功能

| 接口 | 说明 |
|---|---|
| `POST /api/v1/chat/stream` | 多轮对话，SSE 流式逐 token 推送。CH02 起挂 Function Calling：模型自选工具 → 执行 → 回灌收敛，工具执行段推 `tool_start`/`tool_end` 状态帧，全量聊天记录（含工具轨迹）落 MySQL |
| `POST /api/v1/chat/extract` | 售后描述结构化抽取：`order_id` / `claim_type`（八枚举）/ `expected_solution`，未提及为 `null` |
| `GET /` | 聊天页（原生 HTML/JS）：消息流、流式渲染、工具轨迹徽章、可点引用角标（点开看来源章节与原文）、每条回答的 👍/👎 反馈 |

**RAG 进阶**（CH04）：

- **混合检索**：关键词路落在 Milvus 原生 BM25 上（`text` 字段挂 BM25 函数，中文用内置 jieba 分词），向量路 1024 维 COSINE；两路各召回 Top-50，用 `hybrid_search` 配 RRF（k=60）融合。
- **重排**：`bge-reranker-v2-m3` 对粗排 Top-20 精排出 Top-10；喂进 prompt 时按「最相关的放首尾」排列，避开中段注意力低谷。
- **元数据过滤**：`category` 作为 Milvus 标量字段，先过滤再检索。
- **Query 理解**：口语问法先归一回标准问法（LLM，失败回落原句），同义词只在检索侧扩展（「邮费」→「运费」），入库侧一字不改。
- **引用溯源**：证据在 prompt 里带 `[n]` 编号，编号与下发给前端的 citation 列表严格同源；答案里的角标点开即显示来源章节路径与原文。
- **拒答与低置信度池**：检索零命中、或生成前证据自评判「不够答」→ 明确拒答，并把原话落进 `low_confidence_questions`（来源/原因/会话）。闸门在生成之前——答复是流式的，吐出去再判就只剩事后道歉。
- **评估体系**：24 条带难度梯度与 ground-truth 的分桶用例 + 2 条不可回答用例；检索段出 Recall@5/@10、MRR 的四策略对比，生成段用 LLM 裁 Faithfulness，判编造的写进 `faith_cases` 台账（一题一行、复发退回未解决）。

**五个工具**（`app/core/tools/business_tools.py`）：`query_order` / `query_product` / `query_logistics`（工具内 mock 随机生成，同入参结果确定）、`query_faq`（CH03 起内部为向量语义检索：BGE-M3 嵌入 → Milvus Top-5 → 回 MySQL 取原文；向量库故障时降级 SQL LIKE 兜底）、`create_ticket`（写 tickets 表）。工具基础设施：注册管理、参数 Schema 校验、错误分诊（参数不合法 / 查询落空 / 真故障）、超时控制、暂时性故障重试框架。

**RAG 知识库**（`app/rag/`，CH03）：

- **离线建库**：Markdown 知识文档按标题层级结构感知切分（超长递归切、块间重叠裁到最近句号、大表格按行切且每块复制表头）；`category + questions + answer` 三格拼成向量化文本，章节路径 / 内容类型 / 关键条款标记 / 前后块指针四类元数据只存不进向量。
- **双写落库**：MySQL `knowledge_chunks` 是原文权威源，Milvus `knowledge` 集合存向量（主键与 chunk id 严格对齐）。先写 MySQL 记 `pending`，嵌入后 upsert Milvus 回填 `vector_id` 转 `done`；按主键幂等，中断后重跑只补 pending。
- **对话挖知识**：`app/rag/miner.py` 把历史客服对话分批喂 LLM 抽 QA 对，先进 `qa_extraction_staging` 暂存、整体去重（规范化问法）后入知识库；服务端定时开关 `MINE_QA_ENABLED`（默认关）。
- **在线检索**：问题向量化 → Milvus 相似度 Top-K → 回 MySQL 取原文，替换 `query_faq` 的关键词实现（工具入参出参契约不变）。

**多轮上下文**：历史落 MySQL `messages` 表，按 token 预算裁剪（默认 2048，只裁历史，system prompt 恒在首位）；同一 `session_id` 连续发消息复用最近进行中的会话，未知 `conversation_id` 静默走复用/新建。

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
python -m pytest                            # 231 项单测（SQLite 内存库覆盖数据层）
python -m tests.eval.run_eval.py            # CH01 抽取 eval，13 条（需真实 key）
python -m tests.eval.run_tool_eval.py       # CH02 工具选择 eval，13 条（需真实 key）
python -m tests.eval.run_recall_eval.py     # CH03 换说法召回 eval，14 条（需真模型+真库）
python -m tests.eval.run_rag_eval           # CH04 四策略检索 + Faithfulness + 拒答（需真模型+真库）
```

eval 阈值 90%。最近真实运行：CH01 抽取 12/13 = 92%；CH02 工具选择 **13/13 = 100%**（连跑两轮稳定）；CH03 换说法召回 **14/14 = 100%**（真 BGE-M3 + Milvus，Top-5 命中判定）。CH04 四策略 × 分桶（26 题，24 可答 + 2 不可答，真 BGE-M3 + 真重排）：

| 策略 | R@5 | R@10 | MRR |
|---|---|---|---|
| dense | 1.000 | 1.000 | 0.896 |
| bm25 | 0.833 | 1.000 | 0.734 |
| hybrid | 0.958 | 1.000 | 0.882 |
| **hybrid_rerank** | **1.000** | **1.000** | **0.913** |

拒答 2/2（两条知识库没有的问题，证据各 10 条但被判「答非所问」）。注意 hybrid 不重排时 MRR 反而低于纯 dense——RRF 把 BM25 弱路的口语桶噪声摊了进来，这也是默认策略定 `hybrid_rerank` 的实测依据。报告见 `reports/ch04_retrieval.md`。

## 关键配置（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/` | OpenAI 兼容上游地址 |
| `LLM_MODEL` | `glm-5.3-flash` | 模型名 |
| `LLM_API_KEY` | — | 密钥，必填 |
| `LLM_STRUCTURED_METHOD` | `json_mode` | 抽取的结构化输出方式 |
| `DATABASE_URL` | `mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent` | 数据库连接（3307 避让本机 3306） |
| `MYSQL_ROOT_PASSWORD` | — | docker-compose 的 root 密码 |
| `DB_ECHO` | `false` | SQL 日志开关 |
| `TOOL_TIMEOUT_SECONDS` | `10.0` | 工具执行超时 |
| `HISTORY_TOKEN_BUDGET` | `2048` | 历史消息 token 预算 |
| `VECTOR_STORE_URI` | `./data/milvus_lite.db` | 向量库地址（不能叫 `MILVUS_URI`，与 pymilvus 内部配置撞名） |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型（本地目录或 HF 名） |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型（本地目录或 HF 名，~2.3G） |
| `EVIDENCE_SELFCHECK_ENABLED` | `true` | 生成前的证据自评闸；关掉则只按「零命中」拒答 |
| `MINE_QA_ENABLED` | `false` | 对话挖知识定时任务开关 |
| `MINE_QA_INTERVAL_HOURS` | `24` | 挖知识定时间隔 |
| `MINE_QA_BATCH_SIZE` | `20` | 每批挖掘的会话数 |

## 已知限制

1. 只做**单轮**工具调用：模型一次定工具（可并行多个）→ 执行 → 回灌收敛；多轮自动循环的 Agent Loop 是 CH05。
2. CH04 起默认走 **hybrid_rerank**；dense 与 BM25 的分数不同量纲、RRF 分只反映名次，跨策略只能比 Recall/MRR，不能比分值。LIKE 降级路仍保留为向量库故障时的兜底。
3. 向量库存 Milvus Lite 文件库（单机嵌入式，无 Docker 依赖）：索引定死 FLAT 暴力精确检索——一是演示量级（几百块）暴力扫更准，二是 milvus-lite 的 AUTOINDEX 走 faiss，其 Windows 窄字符文件 API 写不了中文路径，跨进程重开必崩（实踩，见 dev-notes/ch03.md）；集合存主键+向量+text(供 BM25 建索引)+category(供过滤)，原文正文一律回 MySQL 取。
4. 挖知识的 QA 去重只做规范化精确/包含比对，语义去重仍待补；挖出的 QA 用固定分类标签「对话挖掘」，类目归并入 CH09 飞轮。
5. 引用编号只保证「模型用了返回里出现过的编号」，不保证编号与句子语义严格对应（模型可能把 [2] 标在 [1] 的内容后）；要强约束得做逐句归因，留给后续。
6. mock 三工具（订单/商品/物流）函数内随机生成，不接真实系统（时间基准为固定常量，同入参跨天调用逐字一致）；`create_ticket` 工单号当日序列生成，撞号有 `IntegrityError → seq+1` 重试兜底，但取号本身非原子，极端并发下仍可能连续撞号（CH08 权限与审计章前以注释标注）。
7. 无鉴权，仅限本地/内网开发用；会话随服务重启仍在（DB 持久化），但无用户体系。
8. GLM-5.3-Flash 思考模式不可关，首 token 延迟高于纯非思考模型。
9. BM25 中文分词用 Milvus 内置 **jieba**（服务端的 `chinese`/lindera 分词器未编进 milvus-lite）；换 Milvus 服务端后建议改回 `chinese`，代码里只改 `ANALYZER` 一处。
10. SSE `data` 为 JSON，中文走 `\u` 转义（前端解析无影响；CH08 工具结果序列化时统一处理中文不转义）。
11. 拒答闸以「知识库」为口径：同一轮里既查单/查物流又查知识库时，只要 `query_faq` 零命中就整轮拒答（连同轮的工具结果一起不答），保守取向——宁可少答一次也不让模型凭常识补政策。反方向放行会让「政策类问题」失去唯一的硬约束。
12. 证据自评失败按「证据够」放行（fail-open），与范本的 fail-closed 相反：CH04 尚无转人工出口，自评抖动时全员拒答等于服务停摆；CH05 接上转人工/建工单后改为 fail-closed。

设计文档：`docs/superpowers/specs/`（ch01/ch02/ch03/ch04 各一份）；实施计划：`docs/superpowers/plans/`；开发留痕：`dev-notes/ch01.md` ~ `dev-notes/ch04.md`。
