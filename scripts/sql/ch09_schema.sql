-- =============================================================
-- ch09 · 数据飞轮 + 可观测（本地替代）· 建表 DDL
-- 新建：review_queue / eval_runs / intent_usage
-- 并给 ch04 的 low_confidence_questions 加两列：
--   retrieved_chunks 落池时的召回片段快照
--   matched_review_id 查重后归并到的缺口
-- 建表顺序：先 review_queue，再加指向它的外键
-- 注意：ALTER 语句不幂等，重复执行会报「列已存在」；重跑前先看表结构
-- =============================================================

SET NAMES utf8mb4;

-- 待审队列：一行 = 一个去重后的知识缺口；查重命中只累加 occurrence_count
CREATE TABLE IF NOT EXISTS review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_question_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案,走 ch03 落库流程写回知识库',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

-- 评估轮次：一行 = 评估流水线跑完的一轮，各指标收进 metrics JSON
CREATE TABLE IF NOT EXISTS eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的:定时任务,或某次改动后手动跑',
  dataset_size INT UNSIGNED    NOT NULL                COMMENT '这轮跑的评估集条数',
  metrics      JSON            NOT NULL                COMMENT '各指标分数,如 {"recall_at_5":0.84,"mrr":0.664,"faithfulness":0.90}',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

-- 按意图的 token 花销（需求 2 的本地替代；Langfuse 未接入时靠它回答「哪类意图烧钱」）
CREATE TABLE IF NOT EXISTS intent_usage (
  id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '流水主键',
  conversation_id   BIGINT UNSIGNED NULL                    COMMENT '所属会话,会话删了置空但不删流水',
  intent            VARCHAR(32)     NOT NULL                COMMENT '意图识别结果(八分类之一)',
  route             VARCHAR(32)     NULL                    COMMENT '路由(知识库/业务数据/退款/人工/其他)',
  prompt_tokens     INT UNSIGNED    NOT NULL DEFAULT 0      COMMENT 'prompt 侧 token',
  completion_tokens INT UNSIGNED    NOT NULL DEFAULT 0      COMMENT '答复侧 token',
  token_source      ENUM('estimated','api') NOT NULL DEFAULT 'estimated' COMMENT 'estimated=本地估算,api=上游 usage 回传',
  model             VARCHAR(64)     NULL                    COMMENT '当轮生成模型名',
  created_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '落表时间',
  PRIMARY KEY (id),
  KEY idx_intent (intent),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_usage_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='按意图的 token 花销流水';

-- 原话流水记归并落点：标准化查重后指向 review_queue 里的缺口行，NULL 表示尚未处理
-- 再存一份落池当时的召回片段快照（Top 几条的原文和得分），没走检索的入口为 NULL
ALTER TABLE low_confidence_questions
  ADD COLUMN retrieved_chunks  JSON            NULL COMMENT '落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL' AFTER reason,
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER retrieved_chunks,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL;
