# 电商智能客服系统

电商客服 Agent 系列·第 1 章：纯对话。FastAPI + LangChain，模型接入走 OpenAI 协议（当前配置为智谱 GLM-5.3-Flash，换供应商只改 `.env`）。

## 功能

| 接口 | 说明 |
|---|---|
| `POST /api/v1/chat/stream` | 多轮对话，SSE 流式逐 token 推送（`event: token` → `event: done`，出错 `event: error`） |
| `POST /api/v1/chat/extract` | 售后描述结构化抽取：`order_id` / `claim_type`（八枚举）/ `expected_solution`，未提及为 `null` |

多轮上下文：服务端内存 session，历史按 token 预算裁剪（默认 2048，只裁历史，system prompt 恒在首位）。

## 快速开始

```bash
# 1. 装依赖（任意 venv）
pip install -r requirements.txt

# 2. 配置
cp .env.example .env   # 填入 LLM_API_KEY（智谱开放平台密钥）

# 3. 起服务
python -m uvicorn app.main:app --port 8000
```

## 验收命令

```bash
# 流式对话（逐 token）
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "你好，我想问下退货政策"}'

# 两轮上下文（第二轮问回订单号）
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-2", "message": "记住我的订单号是 A12345"}'
curl -sN -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-2", "message": "我刚才说的订单号是什么？"}'

# 结构化抽取
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "订单号20241015001的鞋子穿两天开胶了，我要退货退款"}'
```

## 测试与评测

```bash
python -m pytest                       # 36 项单测
python tests/eval/run_eval.py          # 13 条标注样例 eval（需真实 key）
```

eval 阈值 90%（`PASS_THRESHOLD`），匹配规则：`order_id`/`claim_type` 精确匹配，`expected_solution` 包含匹配。最近一次真实运行 **12/13 = 92%**（唯一 FAIL 为两可边界样例）。

## 关键配置（.env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4/` | OpenAI 兼容上游地址 |
| `LLM_MODEL` | `glm-5.3-flash` | 模型名 |
| `LLM_API_KEY` | — | 密钥，必填 |
| `LLM_STRUCTURED_METHOD` | `json_mode` | 结构化输出方式，换支持 function calling 的供应商可改 `function_calling` |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `1` | 上游超时与重试 |
| `HISTORY_TOKEN_BUDGET` | `2048` | 历史消息 token 预算 |

## 已知限制

1. 会话存进程内存，服务重启即丢，无持久化。
2. 不设 session 数量与单 session 长度上限（预算裁剪只限 token，不限条数暴增）。
3. GLM-5.3-Flash 思考模式不可关，首 token 延迟高于纯非思考模型。
4. 无鉴权，仅限本地/内网开发用。
5. 抽取准确率依赖模型表现，边界 case 见 `tests/eval/after_sales_cases.jsonl`。

设计文档：`docs/superpowers/specs/2026-10-15-ch01-pure-chat-design.md`；实施计划：`docs/superpowers/plans/2026-10-15-ch01-pure-chat.md`；开发留痕：`dev-notes/ch01.md`。
