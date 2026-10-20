"""日志初始化。

为什么要单独一个文件:uvicorn 只给 `uvicorn.*` 那几个 logger 配了 handler,
项目自己的 logger 记录会因为根 logger 没 handler 而被丢掉——**日志代码写了却
一条不打**,查起来很费时间(本章踩过:节点的 trace 打了但终端里看不到)。

这里把根 logger 的级别设到 INFO 并补一个 StreamHandler,已有 handler 就不重复加,
不会和 uvicorn 的格式打架。级别可用 LOG_LEVEL 覆盖。
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s | %(message)s"))
        root.addHandler(handler)
    # 上游 SDK 的 INFO 太吵,只留警告
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
