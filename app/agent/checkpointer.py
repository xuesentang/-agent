"""图状态的持久化:用一个进程内 checkpointer,线程 id = conversation_id。

为什么是内存版:框架自带两个 checkpointer,`InMemorySaver` 零依赖,
`SqliteSaver` 要额外装 `langgraph-checkpoint-sqlite`。本章不为了一个功能去动
运行环境(装依赖是环境变更),所以先用内存版,把「状态跟着会话走、同一 thread
第二轮能看见第一轮」这件事跑通。

丢了什么:进程重启后图状态清空。但**对话本身不丢**——历史以 MySQL 的
messages 表为权威源,每轮由服务层读出、裁剪后重新注入 State。所以重启后
用户接着聊,体验上无感;丢的只是图里的中间态(上一轮的意图、证据缓存),
下一轮本来就该重算。

要跨进程持久化时,把这里换成 SqliteSaver 即可,节点与服务层都不用改——
它们只认 `checkpointer` 这个接口。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    logger.info("agent checkpointer: InMemorySaver(进程内,重启清空)")
    return InMemorySaver()
