-- CH10 模型微调:主题分类结果表
-- 只由旁路批量任务写;实时对话主链路不碰这张表。
-- question_id 唯一:一个问题只归一次类,重复跑更新标签、不新增行。
-- 挂外键:归类结果离开问题就没有意义(这点和审计表相反)。

CREATE TABLE IF NOT EXISTS topic_classifications (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    question_id   BIGINT UNSIGNED NOT NULL COMMENT '指向 low_confidence_questions.id',
    labels        JSON            NOT NULL COMMENT '多标签,17 类权威类目里命中的若干个',
    classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_question_id (question_id),
    KEY idx_classified_at (classified_at),
    CONSTRAINT fk_topic_question FOREIGN KEY (question_id)
        REFERENCES low_confidence_questions (id) ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '主题分类结果(CH10)';
