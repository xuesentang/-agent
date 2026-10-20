# CH04 · RAG 进阶实施计划

日期:2026-10-20 | 依据 `docs/superpowers/specs/2026-10-20-ch04-rag-advanced-design.md`

| 任务 | 内容 | 验收 |
|---|---|---|
| T1 | 可行性探路:milvus-lite 是否支持 BM25 函数 / hybrid_search / 中文分词 | 跑通建集合→insert→BM25 命中→RRF 融合的闭环 |
| T2 | 向量库 schema v2(text/sparse/category)+ FakeMilvus 升级 | `tests/test_vectorstore.py` 覆盖 dense/sparse/hybrid/过滤 |
| T3 | 建库回填 text/category;检索四策略 + 元数据过滤 | `tests/test_retriever.py` 四策略 + 过滤用例通过 |
| T4 | 重排器(真模型 + Fake)+ Query 理解(同义词/归一) | 单测 + 真栈粗排 Top-20 精排 Top-10 生效 |
| T5 | 引用组装(编号/首尾放置/证据文本)+ 工具改造 | `tests/test_citations.py` + query_faq 输出带编号 |
| T6 | 生成前质量闸 + 两入口入池 + 拒答话术;System Prompt 引用规则与负面知识 | `tests/test_tool_chat_service.py` 四条新用例通过 |
| T7 | 评估体系:分桶用例 + 四策略指标 + 忠实度裁判 + 编造台账 | 能跑出四策略对比数字;不可回答题正确拒答 |
| T8 | 前端:引用角标可点 + 来源面板 + 👍/👎 一次性锁定;README/dev-notes;提交 | 页面手测通过,章验收标准 4/4 |

## 验收标准对照(提示词第四章)

1. 四策略对比报告能跑出数字 → T7 `run_rag_eval.py`,出 Recall@K/MRR 分桶表;
2. 问带具体型号的问题,BM25 那路能命中 → 用例 B01-B06(宽楦/断底/开线/税号/到货提醒/防伪);
3. 答案引用编号能定位回原文,聊天页点引用看来源 → T5 + T8(`citations` 事件 + 右侧面板);
4. 问知识库没有的内容得到明确拒答且入池 → T6(零命中 / 自评不够两条路)+ 用例 U01/U02。

## 风险

- 重排模型 2.27G:下载需校验 safetensors 偏移(已踩过一次截断);加载慢,靠 lifespan 预热。
- 两路召回都返回同一批块时 RRF 分会被放大,分数不可跨策略比较——报告里只比名次类指标。
