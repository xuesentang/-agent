# CH04 · RAG 进阶(混合检索 + 重排 + 评估 + 引用)设计

日期:2026-10-20 | 范围:第四章提示词全部功能需求

## 决策点

**D1 检索四策略统一入口**
`retrieve(strategy=dense|bm25|hybrid|hybrid_rerank)`,生产默认 `hybrid_rerank`。
评估要横向比四种策略,必须让它们共用同一条回表/组装链路,否则分数差异里混着实现差异。

**D2 BM25 分词器用 jieba(不是 chinese)**
milvus-lite 3.2.2 未编入 lindera,`analyzer_params={"type":"chinese"}` 直接报
`unknown tokenizer type: 'chinese' (supported: 'standard','jieba')`;jieba 同为 Milvus
内置中文分词器,且词典更贴电商口语。需 `pip install jieba`。技术选型(原生 BM25 +
hybrid_search RRF)不变,只换同类分词器——已在可行性探路阶段实测跑通才落码。

**D3 融合用原生 hybrid_search + RRFRanker(k=60)**
两路各召回 Top-50(演示量级全库也就几十块,Top-50 等价于不截断,但保持与生产同构)。

**D4 重排 bge-reranker-v2-m3,粗排 Top-20 → 精排 Top-10**
交叉编码器 O(候选) 次前向,不能全库跑;懒加载 + 进程内单例,lifespan 预热。

**D5 元数据过滤在 Milvus 侧做**
`category` 作标量字段,表达式 `category == "..."` 先过滤再检索;
不在 Python 里检索后过滤——那样 Top-K 会被过滤结果吃掉。

**D6 同义词只在检索侧扩展**
入库侧拆存多份会让同一块知识出现多条向量、去重失效、引用编号指回同一段原文却编成多个号。
BM25 查询串 = `canonical + 扩展词`,dense 查询用 `canonical`。

**D7 编号只在一处产生**
`build_citations` 一次生成 n=1..K,prompt 与 SSE 下发共用同一份列表;`order_for_prompt`
只调喂给模型的位置(首尾放置)不重编号——编号是前端回原文的唯一线索。

**D8 质量闸放在生成之前**
答复是流式的,生成完再判只能事后道歉。两入口:
零命中 → `retrieval_low_conf`;LLM 证据自评判不够 → `self_check`;都入 `low_confidence_questions`。
自评本身失败(fail-open)按「够」处理,不误杀。

**D9 引用走旁路 ContextVar,不进工具出参契约**
`query_faq` 返回值仍是给模型读的文本(带编号的证据块);引用结构经
`CitationBucket` 交给服务层,`done` 之前发 `citations` 事件。
桶存可变对象:同步工具在线程池里执行,`set()` 的新值回不到调用方。

**D10 评估:分桶 + 四策略 + 忠实度裁判 + 台账**
24 条可回答(A_policy/B_model/C_colloquial/E_multi)+ 2 条不可回答;
检索段 Recall@5/@10/MRR 分桶出表;生成段 Faithfulness 由 LLM 裁定,判编造的写
`faith_cases`(一题一行,复发退回未解决);不可回答题走生产同一条闸看是否拒答。

## 本章不做

- 指代消解、多轮改写(第五章);
- 语义级去重、意图识别(后续章节);
- 用户反馈入库(本章只做前端采集,`user_feedback` 入口留给数据飞轮章)。
