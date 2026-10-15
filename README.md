# 电商智能客服系统

电商客服 Agent 系列：CH01 纯对话 + CH02 Function Calling 工具链。FastAPI + SQLAlchemy(Async) + LangChain，模型接入走 OpenAI 协议（当前配置为智谱 GLM-5.3-Flash，换供应商只改 `.env`）。

## 功能

| 接口 | 说明 |
|---|---|
| `POST /api/v1/chat/stream` | 多轮对话，SSE 流式逐 token 推送。CH02 起挂 Function Calling：模型自选工具 → 执行 → 回灌收敛，工具执行段推 `tool_start`/`tool_end` 状态帧，全量聊天记录（含工具轨迹）落 MySQL |
| `POST /api/v1/chat/extract` | 售后描述结构化抽取：`order_id` / `claim_type`（八枚举）/ `expected_solution`，未提及为 `null` |
| `GET /` | 聊天页（原生 HTML/JS）：消息流、流式渲染、工具轨迹徽章 |

**五个工具**（`app/core/tools/business_tools.py`）：`query_order` / `query_product` / `query_logistics`（工具内 mock 随机生成，同入参结果确定）、`query_faq`（SQL LIKE 查 faq 表）、`create_ticket`（写 tickets 表）。工具基础设施：注册管理、参数 Schema 校验、错误分诊（参数不合法 / 查询落空 / 真故障）、超时控制、暂时性故障重试框架。

**多轮上下文**：历史落 MySQL `messages` 表，按 token 预算裁剪（默认 2048，只裁历史，system prompt 恒在首位）；同一 `session_id` 连续发消息复用最近进行中的会话，未知 `conversation_id` 静默走复用/新建。

## 快速开始

```bash
# 1. 装依赖（Python 3.13）
pip install -r requirements.txt

# 2. 配置
cp .env.example .env   # 填入 LLM_API_KEY（智谱开放平台密钥）

# 3. 起数据库（Docker MySQL 8.4，主机端口 3307）
docker compose up -d

# 4. 建表 + 灌种子数据（AI 合成的 FAQ，注明见 scripts/seed_faq.py）
python scripts/apply_schema.py
python scripts/seed_faq.py

# 5. 起服务
python -m uvicorn app.main:app --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

## 验收命令（CH02 三条标准）

```bash
# ① 问订单物流：模型选中工具（气泡徽章）并按结果作答
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "订单1001的物流到哪了"}'
# 期望: tool_start(query_logistics) → tool_end → token… → done

# ② 问退货政策：query_faq 查得到并作答
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-2", "message": "退货政策是什么"}'

# ③ 问「邮费是多少」：LIKE 查不出来（预期漏召回，CH03 向量检索升级钩子）
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-3", "message": "邮费是多少"}'
```

## 测试与评测

```bash
python -m pytest                       # 88 项单测（SQLite 内存库覆盖数据层）
python -m tests.eval.run_eval.py       # CH01 抽取 eval，13 条（需真实 key）
python -m tests.eval.run_tool_eval.py  # CH02 工具选择 eval，13 条（需真实 key）
```

eval 阈值 90%。最近真实运行：CH01 抽取 12/13 = 92%；CH02 工具选择 **13/13 = 100%**（连跑两轮稳定）。

## 关键配置（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/` | OpenAI 兼容上游地址 |
| `LLM_MODEL` | `glm-5.3-flash` | 模型名 |
| `LLM_API_KEY` | — | 密钥，必填 |
| `LLM_STRUCTURED_METHOD` | `json_mode` | 抽取的结构化输出方式 |
| `DATABASE_URL` | `mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent` | 数据库连接（3307 避让本机 3306） |
| `MYSQL_ROOT_PASSWORD` | — | docker-compose 的 root 密码 |
| `DB_ECHO` | `false` | SQL 日志开关 |
| `TOOL_TIMEOUT_SECONDS` | `10.0` | 工具执行超时 |
| `HISTORY_TOKEN_BUDGET` | `2048` | 历史消息 token 预算 |

## 已知限制

1. 只做**单轮**工具调用：模型一次定工具（可并行多个）→ 执行 → 回灌收敛；多轮自动循环的 Agent Loop 是 CH05。
2. 「邮费是多少」这类换说法问题 LIKE 查不出——刻意保留的漏召回样例，CH03 换向量语义检索后应能召回。
3. mock 三工具（订单/商品/物流）函数内随机生成，不接真实系统（时间基准为固定常量，同入参跨天调用逐字一致）；`create_ticket` 工单号当日序列生成，撞号有 `IntegrityError → seq+1` 重试兜底，但取号本身非原子，极端并发下仍可能连续撞号（CH08 权限与审计章前以注释标注）。
4. 无鉴权，仅限本地/内网开发用；会话随服务重启仍在（DB 持久化），但无用户体系。
5. GLM-5.3-Flash 思考模式不可关，首 token 延迟高于纯非思考模型。
6. SSE `data` 为 JSON，中文走 `\u` 转义（前端解析无影响；CH08 工具结果序列化时统一处理中文不转义）。

设计文档：`docs/superpowers/specs/2026-10-15-ch02-tools-design.md`；实施计划：`docs/superpowers/plans/2026-10-15-ch02-tools.md`；开发留痕：`dev-notes/ch01.md`、`dev-notes/ch02.md`。
