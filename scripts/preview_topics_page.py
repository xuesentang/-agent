"""只跑主题分布页的迷你服务（开发台用）。

    python scripts/preview_topics_page.py --port 8123
    # 打开 http://127.0.0.1:8123/admin/topics/

## 为什么不直接 `uvicorn app.main:app`

主应用启动时会预热 RAG(嵌入模型 + 重排模型,约 4G 内存)。只为看一张分布图把 4G
顶起来,在小内存机器上会把系统挤崩。这里只挂 `app.api.topics` 的只读路由和静态目录,
不 import 检索器、不加载分类器。

页面的正式入口仍然是主应用的 `/admin/topics`——这个脚本只是「看一眼」的旁路,
路由对象和静态目录都直接取自应用本体,不存在两份实现。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="主题分布页预览服务")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    from app.api.topics import router as topics_router

    app = FastAPI(title="主题分布（预览）")
    app.include_router(topics_router)
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

    print(f"主题分布页：http://{args.host}:{args.port}/admin/topics/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
