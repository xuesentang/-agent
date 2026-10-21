-- CH07 会话上下文管理:会话摘要表 + conversations 两个锚点
-- 锚点只挪 id、不搬数据;摘要一段一条、只追加不重写。

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    conversation_id BIGINT UNSIGNED NOT NULL,
    seq             INT             NOT NULL DEFAULT 1 COMMENT '第几段,同会话内递增',
    content         TEXT            NOT NULL COMMENT '梗概正文(只提炼事实与诉求)',
    from_msg_id     BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '这段覆盖的起始 messages.id',
    to_msg_id       BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '这段覆盖的结束 messages.id',
    tokens          INT             NOT NULL DEFAULT 0 COMMENT '估算 token',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_summary_conv (conversation_id),
    CONSTRAINT fk_summary_conv FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '会话摘要分段(CH07)';

ALTER TABLE conversations
    ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '摘要覆盖到哪条 messages.id(CH07)',
    ADD COLUMN layer1_from_msg_id  BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '层 1 从哪条 messages.id 起(CH07)';
