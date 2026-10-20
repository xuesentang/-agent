"""请求级上下文:工具执行期间可读取当前会话 id / 回写本轮引用。

create_ticket 的 DDL 要求 conversation_id NOT NULL,但这个内部 id 不能进
工具 schema(模型没见过它会瞎编)。用 ContextVar 在服务层设值、工具层读取,
executor 与工具 schema 都不需要感知它。

引用桶也用 ContextVar,但存的是**可变对象**:同步工具会被 LangChain 丢进线程池
执行,线程里 set 的新值回不到调用方(上下文是复制出去的),而原地修改共享对象
两边都看得见。所以这里只允许改 bucket,不允许重新 set 一个 bucket。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)


@dataclass
class CitationBucket:
    """本轮检索命中的证据(引用编号列表)。

    called 区分「没走知识库」和「走了但零命中」——前者不做证据自评,
    后者要拒答并入低置信度池。
    """

    called: bool = False
    items: list = field(default_factory=list)

    def fill(self, citations: list) -> None:
        self.called = True
        self.items = list(citations)


current_citations: ContextVar[CitationBucket | None] = ContextVar(
    "current_citations", default=None
)
