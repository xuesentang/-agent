# 第三章 RAG 设计

## 目标与验收

保留现有 `query_faq(keyword)` 的输入和 `{found, matches}` 输出，让“邮费是多少”召回运费说明并正确作答。离线建库故意在 MySQL 提交后、Milvus 回填前中断，重跑后应补齐且不产生重复向量。

## 边界

沿用本机 MySQL80、现有 DeepSeek 配置和聊天入口。仅使用 BGE-M3 的 dense 向量及 Milvus COSINE Top-K；不做关键词混合召回、重排或多轮 Agent Loop。`query_faq` 只改内部检索，聊天 SSE、工具名称及参数保持不变。Windows 原生环境无法直接运行 Milvus Lite；使用已确认的 Zilliz Cloud Free 集群作为 Milvus 服务，不启动 Docker。云端只保存向量和 MySQL 数字主键，知识原文仍在本机 MySQL。

## 数据流

1. Markdown 先按标题层级解析。章节内超长正文按段落、句子递归切分；块间保留完整句的重叠。表格超过阈值时按行分块，每块复制表头。政策与手册块的 `questions` 取章节标题，`category` 取上级标题路径；FAQ 块用真实问法。
2. 历史对话离线任务按会话分批读取 `messages`，调用现有模型抽取问答，先写 `qa_extraction_staging`；全批抽取完成后按规范化问句与答案整体去重，保留项转为知识块。定时由 Windows 任务计划程序调用项目 CLI，避免 Web 服务重启时重复启动任务。
3. `knowledge_chunks` 是原文权威源。每块先以 `pending` 状态提交 MySQL。向量化文本只包含 `category`、`questions`、`answer`。BGE-M3 生成 dense 向量后，以 MySQL `id` 作为 Milvus `knowledge` 集合主键执行 upsert；成功后回填 `vector_id=str(id)`、状态改为 `done`。
4. 重跑扫描 `pending` 块并再次 upsert 同一主键。即使上次已写 Milvus、尚未回填 MySQL，也只会覆盖同一向量。文档重导入时，消失的旧块先标为 `retired`，检索立即排除，索引任务再删除对应 Milvus 向量并转为 `retired_clean`；删除失败仍保留 `retired` 供重试。在线检索只用 Milvus 取 ID 和相似度，再按 ID 从 MySQL 取 `done` 原文，组装原有 `matches` 结构；Milvus 或嵌入模型异常时显式返回工具错误，不把故障伪装成“没找到”。

## 数据与环境决定

- 参考文件给出的两张 MySQL 表是基础结构。文档重复导入还需要稳定的来源标识与内容指纹；在 `knowledge_chunks` 增加 `source_key` 和 `content_hash`，为两列组合加唯一约束。同一来源重跑时内容不变则复用主键，内容变化则原行改为 `pending` 并在向量库覆盖同一主键。为防止过期政策继续被召回，`vectorize_status` 使用受约束的四态 `pending/done/retired/retired_clean`，不沿用参考 DDL 的两态 ENUM。用户已同意来源标识扩展，四态是审查后为正确处理删除所加的必要调整。
- Zilliz Cloud 使用 Free 集群，而不是 30 天试用额度所支持的 Serverless 或 Dedicated 集群。连接地址和 API token 只放本机 `.env`，绝不提交。演示时需要联网；云服务限额及服务可用性受提供方约束。PyMilvus 的 URI 配置保留可切换到其他 Milvus 实例的能力。
- Milvus 建议仅保存 `id` 与 1024 维向量；原文、章节路径、内容类型、关键条款标记和前后块指针只存 MySQL。模型本机缓存当前只有引用，没有完整权重；首次运行需下载 BGE-M3。默认 CPU 推理，以适配当前 16 GB 内存、Intel Arc 设备。
- 现有 MySQL 四表保持原样；迁移只新建 `knowledge_chunks` 与 `qa_extraction_staging`。实际建表前检查目标库及表结构；用户已确认第三章按此方案继续。

## 验证顺序

切分规则与幂等逻辑先写失败测试，再实现；使用标注问答集验证非代码知识内容。随后做 MySQL/Milvus 真实双写中断恢复、真实 BGE-M3 语义召回、聊天页工具调用与完整离线测试。每个独立步骤完成时追记 `dev-notes/ch03.md`。

## 官方接口核对

- Context7 的 PyMilvus 文档确认 `MilvusClient.create_collection` 可指定 `auto_id=False`、`id_type="int"`、`dimension` 和 `metric_type="COSINE"`，`upsert` 按主键写入，`search` 返回命中 ID。
- Context7 的 FlagEmbedding 文档确认 `BGEM3FlagModel('BAAI/bge-m3')` 的 `encode(..., return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']` 用法。
- Context7 的 SQLAlchemy 2.0 文档确认 `Base.metadata.create_all()` 只创建缺失表，不迁移既有表。
