# 电商客服：第二章 Function Calling

本章在第一章纯对话基础上加入单轮 Function Calling：模型每轮最多选择一个工具，工具结果回灌后生成最终 SSE 流式答复。使用 DeepSeek OpenAI 兼容接口，不做 Agent Loop、RAG 或真实电商 API。

## Windows 启动

项目使用 Python 3.12+。在 PowerShell 中：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install fastapi 'uvicorn[standard]' langchain-core langchain-openai python-dotenv SQLAlchemy PyMySQL pytest httpx
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

已有 `.env` 时不要覆盖。编辑 `.env`，填入 DeepSeek 的 `MODEL_API_KEY` 和本机 MySQL 的 `DATABASE_URL`；`MODEL_BASE_URL`、`MODEL_NAME` 保持现有 DeepSeek 配置。密码不要提交或发给他人。数据库地址格式见 `.env.example`，密码含 `@`、`:`、`/` 等字符时需先做 URL 编码。当前 DeepSeek 配置使用 `STRUCTURED_OUTPUT_METHOD=json_mode`。
`HISTORY_TOKEN_BUDGET` 是单次请求的近似总预算，其中固定预留 512 token 给输出；请将它设置在所用模型的上下文窗口以内。近似计数与上游实际计费 token 可能有偏差。

确认本机 `MySQL80` 服务在运行，使用你自己的 MySQL 账号创建**独立**数据库：

```sql
CREATE DATABASE IF NOT EXISTS customer_service CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

这条命令只创建项目数据库；不会修改已有其他数据库。将 `DATABASE_URL` 指向 `127.0.0.1:3306/customer_service` 后启动：

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m customer_service.seed
.venv\Scripts\python.exe -m uvicorn customer_service.main:_build_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

`seed` 会在项目数据库中创建四张表并幂等写入 FAQ 样例。订单、商品和物流查询返回随机演示数据；FAQ 使用 `faq.question LIKE`；工单写入 `tickets`。会话和消息由本机 MySQL 持久化。运行本项目不需要 Docker。

本章仅供本机单人演示。`conversation_id` 是会话标识，不是身份凭证；接口尚未做用户鉴权或会话所有权校验，不能直接对外开放。

服务启动后，在浏览器打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 即可用测试页进行流式多轮对话。点击“新建对话”会丢弃页面上的当前会话 ID。售后字段提取仅通过后端接口验收，不在聊天页显示。

## curl 验收

在另一 PowerShell 窗口运行。首轮输出的 `session` 事件包含 `conversation_id`；复制其值到第二轮请求。

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"message":"我的订单号是 A123，请记住"}'

curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"conversation_id":"把首轮的 ID 填在这里","message":"刚才我的订单号是什么？"}'

curl.exe -X POST http://127.0.0.1:8000/aftersales/extract -H "Content-Type: application/json" -d '{"description":"订单 A123 的耳机坏了，请退款"}'
```

有工具时预期事件顺序为 `session`、`tool_status(running)`、`tool_status(success/error)`、若干 `token`、`done`；无工具时没有 `tool_status`。页面在助手气泡显示工具徽章。“邮费是多少”按字面关键词预期查不到 FAQ，这是本章已知漏召回。售后结构化接口继续独立提供，不在页面显示。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

离线测试使用假模型，不需要真实密钥。第一章 DeepSeek 结构化提取验收仍保留。

填好 `.env` 后，还可让真实上游跑六条标注样例：

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe evals\run_aftersales.py
```

第二章真实 DeepSeek 与本机 MySQL 验收：

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe evals\run_ch02.py
```

同步工具在工作线程执行；超时不能强制取消已启动的线程，只读工具重试时旧线程可能仍在运行。`create_ticket` 从不自动重试，避免生成重复工单。

售后标注评估会逐条列出实际字段与标注是否一致；有不匹配时退出码为 1。
