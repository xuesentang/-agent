我要用 Superpowers 模式给客服系统建知识库,把 query\_faq 的内部实现从关键词查表升级成向量语义检索,工具的入参出参契约保持不变。



\## 功能需求

1\. 离线建库·文档处理:知识文档(退货政策、商品 FAQ、售后手册这类 Markdown)按标题层级做结构感知切分;超长内容递归切;块间加重叠且裁到最近句号,不留半截话;大表格按行切时每块都复制表头

2\. 离线建库·对话挖知识:做一条从历史客服对话挖知识的定时任务,分批喂给 LLM 抽取问答对,先进暂存表、再整体去重入库

3\. 落库结构:每条知识带 category、questions、answer 三个字段,拼成一段文本做向量化;商品 FAQ 和挖出来的问答对,questions 填真实问法;政策手册这类没有天然问题的,questions 填所在章节标题、category 填上级标题路径;另带章节路径、内容类型、是否关键条款、前后块指针四类元数据,只存不进向量

4\. 双写落库:MySQL 建 knowledge\_chunks 表当原文权威源、Milvus 建 knowledge 集合;先写 MySQL 记「待向量化」,再写 Milvus 拿 vector\_id 回填、状态转「已向量化」;按主键幂等,挂了能重跑

5\. 在线检索:问题向量化后到 Milvus 按相似度取 Top-K,替换掉 query\_faq 的关键词查表实现



\## 技术栈

\- 嵌入模型 BGE-M3

\- 向量库 Milvus,MySQL 当原文权威源



\## 本章不做

\- 关键词召回、混合检索、重排,本章只跑 dense 向量单路



\## 验收标准

1\. 「邮费是多少」这类换说法的问题,现在能召回运费说明并答对

2\. 故意中断建库任务再重跑,漏向量化的块能被捡起补齐



\## 工作要求

1\. 全程走 Superpowers 流程,技能自动触发;产出不是可单测代码的任务(纯 Prompt、数据类),把 TDD 那步换成拿标注样例或评估集跑一遍验证,其余步骤照走

2\. 过程留痕:在仓库 dev-notes/ch03.md 里追记开发过程,每完成一个阶段(brainstorm 定稿、计划评审通过、每个任务完成、code review 结论、finish)就补一段,记四样:我这一步发的关键原话、你的关键产出(spec / plan 路径、评审结论)、我拒绝或纠偏了什么、翻车与返工;不许收尾时一次性补记

3\. 涉及具体库、框架、API 的用法(FastAPI、SQLAlchemy、LangChain、LangGraph、Milvus、Langfuse 这些),一律先用 Context7 MCP 查最新官方文档和接口定义再动手,别凭记忆写,版本对不上的 API 是返工重灾区

4\. 上面点名的技术选型是定死的,实现中发现矛盾或走不通,停下来问我,不要自行换方案

5\. 完结交付:功能演示命令、测试结果、dev-notes 路径                                              







\-- =============================================================

\-- ch03 · RAG 基础 · 建表 DDL

\-- 本章新建:knowledge\_chunks(知识库原文权威源)

\-- 向量落 Milvus 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态

\-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量

\-- =============================================================



\-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)

SET NAMES utf8mb4;



CREATE TABLE knowledge\_chunks (

&#x20; id               BIGINT UNSIGNED NOT NULL AUTO\_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',

&#x20; category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',

&#x20; questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',

&#x20; answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',

&#x20; section\_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',

&#x20; content\_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',

&#x20; is\_key\_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',

&#x20; prev\_chunk\_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',

&#x20; next\_chunk\_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',

&#x20; vector\_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',

&#x20; vectorize\_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',

&#x20; created\_at       DATETIME        NOT NULL DEFAULT CURRENT\_TIMESTAMP COMMENT '创建时间',

&#x20; updated\_at       DATETIME        NOT NULL DEFAULT CURRENT\_TIMESTAMP ON UPDATE CURRENT\_TIMESTAMP COMMENT '更新时间',

&#x20; PRIMARY KEY (id),

&#x20; KEY idx\_category (category),

&#x20; KEY idx\_vectorize\_status (vectorize\_status),

&#x20; CONSTRAINT fk\_chunks\_prev FOREIGN KEY (prev\_chunk\_id) REFERENCES knowledge\_chunks (id) ON DELETE SET NULL,

&#x20; CONSTRAINT fk\_chunks\_next FOREIGN KEY (next\_chunk\_id) REFERENCES knowledge\_chunks (id) ON DELETE SET NULL

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';



CREATE TABLE qa\_extraction\_staging (

&#x20; id               BIGINT UNSIGNED NOT NULL AUTO\_INCREMENT COMMENT '暂存行主键',

&#x20; batch\_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',

&#x20; source\_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',

&#x20; question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',

&#x20; answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',

&#x20; status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',

&#x20; created\_at       DATETIME        NOT NULL DEFAULT CURRENT\_TIMESTAMP COMMENT '抽取写入时间',

&#x20; PRIMARY KEY (id),

&#x20; KEY idx\_batch\_no (batch\_no),

&#x20; KEY idx\_status (status)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge\_chunks,建库完成可清空';

