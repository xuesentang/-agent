# CH03 RAG 知识库（向量语义检索）— 设计 Spec

> 状态：spec 定稿待执行；**唯一卡点 = D2（BGE-M3 接入路线）需用户拍板**，其余按单章连续执行模式推进。
> 依据：`各章提示词.md` 第三章、`各章建表.md` ch03 节、`AGENTS.md`。

## 1. 目标与范围

给 CH02 的 `query_faq` 换上向量语义检索：离线把 Markdown 知识文档 + 历史对话挖出的 QA 对建成知识库（MySQL 存原文权威源，Milvus 存向量），在线把用户问题向量化后到 Milvus 取 Top-K，回答依据召回内容组织。**工具入参出参契约保持不变**（模型侧无感）。

**本章不做**：关键词召回、混合检索、重排（CH04）；上下文压缩/摘要（CH07）；权限审计正式版（CH08）。

## 2. 决策点

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | Milvus 部署 | docker-compose 增加 **Milvus standalone + etcd + minio**（与 MySQL 同套路，compose 一并交付）；数据卷落命名卷 | 提示词定死 Milvus；standalone 是官方推荐的最简完整部署，比 Milvus Lite（pip 文件式）更贴近生产形态、面试可讲；15.6G 内存可承受 |
| D2 | BGE-M3 接入路线 | **待用户拍板**（见 §9）：A 本地推理（torch+sentence-transformers，模型 ~2.2G）／B OpenAI 兼容 embeddings API（需用户供 key）。嵌入适配器接口先定死：`embed_texts(list[str]) -> list[list[float]]`，1024 维，实现可换 | 提示词定死 BGE-M3 模型但未定接入方式；本地重依赖 vs 外部 key 属「依赖选型 + 可能外部资源」，按 AGENTS.md §4 停下问；维度与契约两路线一致，不赌方向 |
| D3 | 向量化文本拼接 | `category + questions + answer` 拼成一段（DDL 注释原话：「三格拼成向量化文本」）；元数据四类（section_path/content_type/is_key_clause/前后指针）只存不进向量 | 建表 DDL 定死 |
| D4 | 切分引擎 | 纯 Python 零新依赖，独立模块 `app/rag/chunker.py`：① Markdown 按标题层级结构感知切分；② 块超上限递归按段落/句子细切；③ 相邻块重叠内容裁到最近句号边界（不留半截话）；④ 大表格按行切、每块复制表头。纯函数 + 数据类，TDD 主战场 | 切分逻辑是本章最密的可单测部分；零依赖保证单测不拽 torch/Milvus |
| D5 | 双写状态机 | 建库脚本先写 MySQL（vectorize_status=pending，拿自增 id）→ 批量 embed → Milvus insert（主键=chunk id）→ 回填 vector_id + status=done。**按主键幂等**：重跑只捞 status=pending 的块补向量化；Milvus insert 用 upsert 语义兜底重复 | 提示词「先写 MySQL 记待向量化、再写 Milvus 回填、挂了能重跑」；DDL 有 idx_vectorize_status |
| D6 | 挖知识流水线 | 定时任务形态 = **CLI 脚本为主**（`scripts/mine_qa.py`，可手跑可挂 cron/计划任务）+ FastAPI lifespan 可选注册 apscheduler（默认关，env 开关）；流程：按 batch_no 分批取 messages 历史 → LLM 抽 QA 对 → 写 `qa_extraction_staging`（status=extracted）→ **整体去重**（规范化后与知识库现有 questions 精确+包含去重）→ kept 项入 knowledge_chunks（content_type=qa_mined）→ discarded 标注 | 提示词「先进暂存表、再整体去重入库」；CLI 优先保证可单测可复跑，apscheduler 作演示开关不引入强制运行时依赖 |
| D7 | query_faq 换实现 | 入参 `keyword: str`、出参文本（命中列表或「未找到」）**不动**；内部：keyword → embed → Milvus search(top_k, 默认5) → 按 id 回 MySQL 取原文 → 格式化成与 CH02 一致的文本。Milvus 不可用时降级：回 MySQL LIKE（CH02 原实现保留为 fallback，日志告警） | 提示词「入参出参契约保持不变」；降级保可用性（面试可讲：检索降级不打断对话） |
| D8 | 测试策略 | 单测：切分引擎全分支（SQLite 覆盖新表仓库层；Milvus 用 fake client 测双写状态机与幂等重跑；嵌入用假 embedder 测流水线）。eval：标注样例集专测「换说法召回」（邮费/运费、保修/质保 类），≥90%。集成验收：真实 Milvus + BGE-M3 + GLM 跑两条提示词标准 | 沿用 CH02 双库跑法；fake client 让状态机单测不依赖 Docker |
| D9 | 种子知识文档 | `scripts/seed_knowledge.py` AI 生成逼真 Markdown：退货政策（多级标题+关键条款）、商品 FAQ（真实问法）、售后手册（含大表格、超长章节），注明 AI 合成；灌 knowledge_chunks 前先过切分引擎 | AGENTS.md 数据规则；文档要故意覆盖切分引擎各分支（表格/超长/多级标题） |
| D10 | 目录结构 | 新增 `app/rag/`（chunker.py 切分 / embeddings.py 嵌入适配器 / vectorstore.py Milvus 封装 / indexer.py 建库流水线 / miner.py 挖知识 / retriever.py 在线检索）；工具层 business_tools.query_faq 改调 retriever | 与现有 core/services 分层一致；RAG 子系统独立目录，CH04 混合检索在此上演进 |

## 3. 架构与数据流

```
【离线建库】scripts/build_knowledge.py
  Markdown 文档 ──▶ chunker 结构感知切分 ──▶ knowledge_chunks(MySQL, pending)
                                                      │
  历史 messages ──▶ miner 分批 LLM 抽 QA ──▶ qa_extraction_staging ──▶ 去重 ──┤
                                                                              ▼
                                              embeddings.embed_texts(BGE-M3) ◀── 捞 pending
                                                      │
                                             Milvus knowledge 集合 insert(主键=chunk id)
                                                      │
                                    回填 vector_id + status=done（幂等，重跑补 pending）

【在线检索】用户问题 ──▶ query_faq(keyword) ──▶ embed(keyword) ──▶ Milvus Top-K
      ──▶ 回 MySQL 取原文 ──▶ 格式化文本 ──▶ 回灌模型组织回答
      （Milvus 不可用 → 降级 MySQL LIKE，保 CH02 行为）
```

## 4. 数据库（ch03 DDL，两张新表）

按 `各章建表.md` ch03 节：`knowledge_chunks`（原文权威源 + 双写状态，DDL 注释明确「chunk 主键与 Milvus 集合主键对齐」）、`qa_extraction_staging`（挖知识中转）。建表脚本落 `scripts/sql/ch03_schema.sql`（与建表语句逐字一致），`apply_schema.py` 支持增量执行（已有 `--if-not-exists` 幂等选项）。ORM 模型与 DDL 一一对应；自增主键用现有 `BigInt`（通用 BigInteger + sqlite 回落，CH02 翻车结论）。

## 5. 切分引擎规格（TDD 主战场）

- **结构感知**：以 Markdown 标题（`#`~`######`）为块边界起点，块继承标题路径；目标块长 300~500 字（可配），超限才细切。
- **递归切**：块超上限 → 先按段落切，段落仍超 → 按句子切（中英文句末标点）。
- **重叠**：相邻块重叠 ~15% 且**必须裁到最近句号**，禁止半截话开头/结尾（提示词原文要求）。
- **表格**：单个表格行数超限 → 按行切块，**每块都复制表头**（Markdown 表格头 + 分隔行）。
- **产出**：`Chunk(section_path, content_type, category, questions, answer, is_key_clause, text_for_embedding)` 数据类；纯函数 `split_document(markdown, config) -> list[Chunk]`。

## 6. 测试与验收映射

1. 「邮费是多少」能召回运费说明 → eval 换说法标注集（≥90%）+ 集成验收现场问。
2. 中断建库再重跑，pending 块补齐 → 单测（fake Milvus：插一半挂掉、重跑补全）+ 集成验收（Ctrl-C / 异常注入后重跑，查 pending=0）。

## 7. Review Focus

1. 双写中途失败的一致性：MySQL pending 与 Milvus 已有向量的对账（重跑后不得出现重复向量或永久 pending）。
2. 幂等键：chunk id 与 Milvus 主键严格对齐，重复 insert 不得报错或产生重复行。
3. 切分重叠边界：不得出现半截话（句号裁切）；表格块必须带表头。
4. 去重口径：规范化（去空白/标点/大小写）后与知识库现有 questions 的去重，防 QA 对重复入库。
5. query_faq 契约不变：CH02 的 SSE 徽章、回灌格式、mock 行为不受影响（CH02 回归全过）。
6. Milvus 不可用降级：LIKE fallback 不抛异常、不中断聊天。

## 8. 已知风险

- BGE-M3 CPU 推理速度：Ultra 5 125H 无独显，批量编码几百块预计分钟级，可接受；在线单条查询 <1s。
- Milvus standalone 三容器资源占用：与 MySQL 共存于 15.6G 内存，需观察；若吃紧可降配 minio/etcd 或改 Milvus Lite（保 API 兼容）。

## 9. D2 待拍板项：BGE-M3 接入路线

| 路线 | 做法 | 成本 | 风险 |
|---|---|---|---|
| A. 本地推理（推荐） | pip 装 torch(CPU)+sentence-transformers；模型经 HF 镜像下载 ~2.2G；`embed_texts` 本地批推理 | 首次 ~4G 下载、CPU 推理分钟级建库 | 零外部 key、完全自包含可复现、面试好讲「端到端自建 RAG」 |
| B. OpenAI 兼容 API | SiliconFlow 等提供 BGE-M3 embeddings 接口，`.env` 加 `EMBEDDING_API_KEY` | 零本地重依赖、快 | 需用户提供 key；引入外部服务依赖（限流/可用性） |

两路线均满足「BGE-M3 + 1024 维」定死约束，适配器接口不变，后续可切换。**推荐 A**：与项目「换供应商只改 .env」的一贯纪律一致的是 B，但求职作品讲「全链路自包含」A 更硬；且本机 14 核 CPU 跑演示量级足够。
