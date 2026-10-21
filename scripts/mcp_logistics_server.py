"""物流查询 MCP Server(第八章需求 6)。

独立进程,Streamable HTTP 传输,默认监听 8121:
    python scripts/mcp_logistics_server.py            # 127.0.0.1:8121/mcp
    python scripts/mcp_logistics_server.py --port 9121

数据与内置 `query_logistics` 同源(`app/core/mock_orders.logistics_track_text`),
不建表、不接真实系统——同一个订单号在「内置版」和「MCP 版」下给出同一份轨迹,
换实现不改变答案。

客户端(客服系统)侧配置:
    MCP_ENABLED=true
    MCP_SERVERS=logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.core.mock_orders import logistics_track_text  # noqa: E402

MCP_NAME = "logistics"
DEFAULT_PORT = 8121

mcp = FastMCP(MCP_NAME)


@mcp.tool()
def query_logistics_track(order_id: str) -> str:
    """查询订单的物流轨迹(最新在上)。

    order_id: 订单号,纯数字,例如 1001
    """
    return logistics_track_text(order_id)


@mcp.tool()
def estimate_delivery(order_id: str) -> str:
    """查询订单的预计送达时间。

    order_id: 订单号,纯数字,例如 1001
    """
    lines = logistics_track_text(order_id).splitlines()
    last = lines[-1] if len(lines) > 1 else ""
    return f"订单 {order_id} 当前最新物流节点:{last.lstrip('- ').strip()}。预计送达时间以快递员电话为准。"


def main() -> int:
    parser = argparse.ArgumentParser(description="物流查询 MCP Server")
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
