# 电商客服：第三章 RAG 开发中

> 第三章 RAG 正在 `feat/rag-ch03` 分支开发。现有对话与工具链的完整演示请使用 `main` 分支；第三章需要先配置 Zilliz Cloud Free 和 BGE-M3 依赖。

现有系统提供单轮 Function Calling 与 SSE 流式答复。第三章将 `query_faq` 改为 BGE-M3 + Milvus 语义检索；不做 Agent Loop 或真实电商 API。

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

`seed` 会在项目数据库中创建原有四张表并幂等写入 FAQ 样例。订单、商品和物流查询返回随机演示数据；第三章 FAQ 工具通过 Milvus 语义检索。工单写入 `tickets`，会话和消息由本机 MySQL 持久化。运行本项目不需要 Docker。

本章仅供本机单人演示。`conversation_id` 是会话标识，不是身份凭证；接口尚未做用户鉴权或会话所有权校验，不能直接对外开放。

服务启动后，在浏览器打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 即可用测试页进行流式多轮对话。点击“新建对话”会丢弃页面上的当前会话 ID。售后字段提取仅通过后端接口验收，不在聊天页显示。

## curl 验收

在另一 PowerShell 窗口运行。首轮输出的 `session` 事件包含 `conversation_id`；复制其值到第二轮请求。

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"message":"我的订单号是 A123，请记住"}'

curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"conversation_id":"把首轮的 ID 填在这里","message":"刚才我的订单号是什么？"}'

curl.exe -X POST http://127.0.0.1:8000/aftersales/extract -H "Content-Type: application/json" -d '{"description":"订单 A123 的耳机坏了，请退款"}'
```

有工具时预期事件顺序为 `session`、`tool_status(running)`、`tool_status(success/error)`、若干 `token`、`done`；无工具时没有 `tool_status`。页面在助手气泡显示工具徽章。第三章配置并建库后，“邮费是多少”应召回运费说明。售后结构化接口继续独立提供，不在页面显示。

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

第二章真实 DeepSeek 与本机 MySQL 验收（仅限 `main` 分支；其邮费漏召回断言不适用于第三章）：

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe evals\run_ch02.py
```

同步工具在工作线程执行；超时不能强制取消已启动的线程，只读工具重试时旧线程可能仍在运行。`create_ticket` 从不自动重试，避免生成重复工单。

售后标注评估会逐条列出实际字段与标注是否一致；有不匹配时退出码为 1。

## 第三章 RAG（开发分支）

先在 Zilliz Cloud 建立 **Free** 集群，复制 Endpoint 和 API Key 到本机 `.env` 的 `MILVUS_URI`、`MILVUS_TOKEN`。不要把密钥提交到 GitHub。向量和整数 ID 存云端；政策原文只在本机 MySQL。运行建库与聊天时需联网。首次运行 BGE-M3 会下载模型权重，并在本机 CPU 推理。若 Hugging Face 下载中断，可从 ModelScope 下载同一 `BAAI/bge-m3` 模型，并将本地模型目录写入 `.env` 的 `BGE_M3_MODEL_PATH`。

```powershell
.venv\Scripts\python.exe -m pip install -e '.[rag,test]'
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m customer_service.knowledge_cli init-db
.venv\Scripts\python.exe -m customer_service.knowledge_cli import knowledge\shipping-policy.md
.venv\Scripts\python.exe -m customer_service.knowledge_cli import-faq
.venv\Scripts\python.exe -m customer_service.knowledge_cli index
.venv\Scripts\python.exe evals\run_ch03.py
```

`index` 从 MySQL 中的 `pending` 行继续运行。中断后再次执行同一命令，会以相同 ID 覆盖 Milvus 向量，再把该行标为 `done`。`query_faq` 保持原来的 `keyword` 入参和 `{found,matches}` 输出，但内部改为 dense 语义检索；云端未配置时会报告工具错误。

历史对话问答抽取是离线任务，需要时运行：

```powershell
.venv\Scripts\python.exe -m customer_service.knowledge_cli extract --batch-size 20
.venv\Scripts\python.exe -m customer_service.knowledge_cli promote
.venv\Scripts\python.exe -m customer_service.knowledge_cli index
```

如需定时，使用 Windows 任务计划程序定期调用上述三条命令。项目不自动创建系统计划任务。已在本机验证 Zilliz 连接、BGE-M3 建库中断恢复、语义召回、浏览器聊天回复和真实模型问答抽取；在其他电脑运行仍需各自配置连接与模型目录。离线检查使用 `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`。
