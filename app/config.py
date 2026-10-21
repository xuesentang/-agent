"""Application settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_timeout: int = 60
    llm_max_retries: int = 1
    # with_structured_output method: "json_mode"(DeepSeek/GLM) 或 "function_calling"。
    # DeepSeek 的 JSON 模式要求提示词里出现 "json" 字样,本项目提示词均满足。
    llm_structured_method: str = "json_mode"
    # DeepSeek 的思考模式默认开启(默认 effort=high),关掉才快。
    # 空字符串 = 不发这个字段(换到不认识的供应商时用)。
    # 注意:思考模式 + tools 时,上一轮的 reasoning_content 必须原样回传,否则 400;
    # 本项目 CH02 的工具循环没做这件事,所以默认 disabled。
    llm_thinking: str = "disabled"
    history_token_budget: int = 2048
    # CH02: database (Docker MySQL on host port 3307, see docker-compose.yml).
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent"
    db_echo: bool = False
    # CH02: tool execution timeout (registry executor).
    tool_timeout_seconds: float = 10.0
    # CH03: RAG — 向量库连接与嵌入模型。
    # 注意:不能叫 milvus_uri(env MILVUS_URI)——pymilvus 自己会读 .env 里这个名字,
    # import 时就按 http(s):// 形式解析,相对路径直接抛 ConnectionConfigException。
    vector_store_uri: str = "./data/milvus_lite.db"
    embedding_model: str = "BAAI/bge-m3"
    # CH04: 检索增强 — 重排模型 + 生成前的证据自评开关。
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 检索证据自评(生成前闸门):关掉就只按「零命中」拒答
    evidence_selfcheck_enabled: bool = True
    # 混合检索两路各召回多少条。语料涨到几百块后 50 等于扫 10%,
    # 会把评估重新变成「大范围扫描」——按库规模调到 20(约 4%)。
    recall_route_limit: int = 20
    # CH04 评测:eval 的裁判模型。留空回落生成模型,报告会显式标注「裁判=生成」。
    # 想让绝对分数可信,必须填成与 llm_model 不同的模型(最好不同厂商),
    # 需要时同时给 judge_base_url / judge_api_key。
    judge_model: str = ""
    judge_base_url: str = ""
    judge_api_key: str = ""
    # CH05: 工作流编排 — 知识类问题的置信度闸阈值与主力 Agent 的停止条件。
    # kb_min_score 是 dense 相似度的粗筛线:实测 357 块库上库内问题 0.49~0.78、
    # 库外 0.38~0.62 且有重叠,阈值只能挡明显离题的,精确判定交给证据自评。
    kb_min_score: float = 0.45
    agent_max_steps: int = 4
    agent_token_budget: int = 6000
    # CH07: 上下文管理 — 预算从窗口倒推,不写死历史长度。
    # 这五项是倒推公式的输入,改任何一个都会改变层 1/层 2 的预算(验收 2 用它演示级联)。
    model_context_window: int = 65536
    max_output_tokens: int = 2000
    max_user_input_tokens: int = 2000
    tool_result_max_tokens: int = 1200
    rerank_top_k: int = 5
    # 层 2 里客服答复只留开头这么多字(用户原话不动,工具结果换成一行标识)
    layer2_assistant_head: int = 40
    # 后台摘要开关与超时(摘要失败不影响当轮回复)
    context_summary_enabled: bool = True
    context_summary_timeout: int = 60
    # CH03: 对话挖知识定时任务(默认关;开了 lifespan 起后台循环,按 interval 跑一轮)
    mine_qa_enabled: bool = False
    mine_qa_interval_hours: int = 24
    mine_qa_batch_size: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
