"""请求级上下文:工具执行期间可读取当前会话 id。

create_ticket 的 DDL 要求 conversation_id NOT NULL,但这个内部 id 不能进
工具 schema(模型没见过它会瞎编)。用 ContextVar 在服务层设值、工具层读取,
executor 与工具 schema 都不需要感知它。
"""

from contextvars import ContextVar

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)
