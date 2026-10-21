-- CH08 工具系统:工具调用审计表
-- 内置工具和 MCP 工具都记一条,被权限拒、被校验拦的调用同样落——
-- 审计要能回答「模型试过什么、被什么挡住」,只记成功等于没记。
-- 不挂外键:审计是旁路,写入不能被引用约束拦住;conversation_id 只建普通索引。

CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    conversation_id BIGINT UNSIGNED NULL COMMENT '所属会话,无会话上下文的调用为 NULL',
    tool_call_id    VARCHAR(64)     NULL COMMENT '模型申请单 id,可对回 messages 流水',
    tool_name       VARCHAR(128)    NOT NULL,
    tool_source     ENUM('builtin','mcp') NOT NULL DEFAULT 'builtin',
    mcp_server      VARCHAR(64)     NULL COMMENT '来源 MCP Server 名,内置工具为 NULL',
    arguments       JSON            NULL,
    result_summary  TEXT            NULL COMMENT '结果摘要,过长截断',
    status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL,
    error_message   VARCHAR(512)    NULL,
    retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '写操作恒为 0',
    duration_ms     INT UNSIGNED    NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_audit_conversation (conversation_id),
    KEY idx_audit_tool (tool_name),
    KEY idx_audit_status (status)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4 COMMENT = '工具调用审计留痕(CH08)';
