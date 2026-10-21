"""售后查询 MCP Server(第八章需求 6)。

独立进程,Streamable HTTP 传输,默认监听 8122:
    python scripts/mcp_aftersale_server.py            # 127.0.0.1:8122/mcp

三个只读工具:查在保、查退货进度、查发票状态。数据由订单号派生(固定 seed),不建表、
不接真实系统。

验收 3「Server 侧新加一个工具、只重启该 Server,客服系统不改代码不重启就能用上」
就是在这个文件里加一个 `@mcp.tool()` 然后重启本进程——见 README 的验收命令。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.core.mock_orders import (  # noqa: E402
    invoice_status_text,
    return_progress_text,
    warranty_info,
)

MCP_NAME = "aftersale"
DEFAULT_PORT = 8122

mcp = FastMCP(MCP_NAME)


@mcp.tool()
def check_warranty(order_id: str) -> str:
    """查询订单商品是否在保修期内以及保修到期日。

    order_id: 订单号,纯数字,例如 1001
    """
    active, expires = warranty_info(order_id)
    state = "在保" if active else "已过保"
    return f"订单 {order_id} 的商品{state},保修到期日 {expires}。"


@mcp.tool()
def query_return_progress(order_id: str) -> str:
    """查询退货/退款的处理进度。

    order_id: 订单号,纯数字,例如 1001
    """
    return return_progress_text(order_id)


@mcp.tool()
def query_invoice_status(order_id: str) -> str:
    """查询发票开具状态。

    order_id: 订单号,纯数字,例如 1001
    """
    return invoice_status_text(order_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="售后查询 MCP Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    # 这一版 SDK 的 host/port 在构造参数里,run() 只接 transport;命令行覆盖写回 settings
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
