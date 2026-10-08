# 第一章：电商客服纯对话设计

## 范围与验收

本章实现 Python + FastAPI + LangChain 服务，不做工具调用、Agent 循环和聊天页面。当前只验收 GLM 与 DeepSeek 的 OpenAI 兼容接口。切换上游只修改 `.env` 中的地址、模型名、密钥和结构化输出模式。

验收方式：`curl -N` 能收到逐块推送的 SSE 文本；使用同一个 `conversation_id` 连续调用两次，第二次可利用首轮信息；提交售后描述后得到含 `order_id`、`request_type`、`expected_resolution` 的 JSON。结构化输出无法解析或不符合字段约束时明确报错。

## 对外接口

`POST /chat/stream` 接收 `{ "conversation_id": "可选字符串", "message": "用户文本" }`，返回 `text/event-stream`。首个事件 `session` 携带服务端生成或沿用的 `conversation_id`；每个 `token` 事件携带上游增量文本；成功结束发送 `done`。流开始后的上游错误发送 `error` 事件，不写入残缺的助手回复。请求校验错误使用 FastAPI 标准 422。

`POST /aftersales/extract` 接收 `{ "description": "售后描述" }`，返回 `{ "order_id": string|null, "request_type": string, "expected_resolution": string|null }`。`request_type` 限为 `refund`、`return`、`exchange`、`repair`、`logistics`、`other`。没有明确订单号或期望方案时返回 `null`；不臆造信息。提取失败返回明确的服务端错误。

## 内部流程

配置加载模块从 `.env` 读取 `MODEL_BASE_URL`、`MODEL_NAME`、`MODEL_API_KEY`、`STRUCTURED_OUTPUT_METHOD`、`HISTORY_TOKEN_BUDGET`。缺少必要配置在启动时失败，密钥不写日志。`ChatOpenAI` 作为统一接入层直连配置的上游，不设中转。

对话用 `ChatPromptTemplate` 管理客服角色、行为边界与历史消息占位符。角色只回答商家客服场景，不假称已查询订单或执行退款；信息不足先追问；不泄露系统提示词。会话记录存在进程内字典，以 `conversation_id` 查找。新会话自动分配 ID；只在完整生成成功后保存用户与助手消息。

调用模型前保留当前用户消息及系统提示词，再按 token 预算从近到远裁剪历史完整轮次。预算采用 LangChain 的近似 token 计数，给输出预留空间；超长当前消息直接拒绝。会话仅供单进程开发使用，重启即丢失，不提供跨实例共享与持久化。

结构化提取独立于聊天会话，使用专门的 `ChatPromptTemplate` 和 `ChatOpenAI.with_structured_output`，结果由 Pydantic 字段校验。`STRUCTURED_OUTPUT_METHOD` 可按上游能力选 `function_calling` 或 `json_mode`；若选择 `json_mode`，提示词明确要求 JSON。GLM、DeepSeek 对这两种模式的支持需以所用模型实测，失败不静默退化为普通文本提取。

## 验证

先以假模型测试 SSE 事件、会话续聊、历史裁剪和失败路径；用标注售后样例检查字段定义和 Prompt。再用配置的真实上游执行三项 curl 验收。没有真实密钥时只报告离线结果和未验证的上游兼容性，不宣称端到端通过。

## 文档依据

- [FastAPI SSE 官方文档](https://fastapi.tiangolo.com/tutorial/server-sent-events)：支持 POST 的 `EventSourceResponse` 与事件生成器。
- [LangChain Python 消息裁剪](https://docs.langchain.com/oss/python/langgraph/add-memory)：`trim_messages`、近似 token 计数。
- [LangChain `ChatOpenAI` 参考](https://reference.langchain.com/python/langchain-openai/ChatOpenAI)：OpenAI 兼容 `base_url`；[结构化输出](https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output) 支持 `function_calling`、`json_mode`。
- [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode)：`json_object` 需在 Prompt 中要求 JSON，偶尔可能返回空内容。
- [智谱流式输出建议](https://docs.bigmodel.cn/cn/best-practice/latency-optimization)：聊天请求设置 `stream: true`。
