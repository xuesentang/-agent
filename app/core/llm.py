"""ChatOpenAI factory driven entirely by settings (spec D1)."""

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def _extra_body(settings: Settings) -> dict | None:
    """DeepSeek 的思考开关。空配置 = 不发这个字段(换别家供应商时不至于报未知参数)。"""
    if not settings.llm_thinking:
        return None
    return {"thinking": {"type": settings.llm_thinking}}


def _base_model(temperature: float) -> ChatOpenAI:
    settings = get_settings()
    kwargs = {}
    extra = _extra_body(settings)
    if extra:
        kwargs["extra_body"] = extra
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,  # review Minor #7: never hang SSE on a silent upstream
        max_retries=settings.llm_max_retries,
        **kwargs,
    )


def get_chat_model() -> ChatOpenAI:
    """Model for the customer-service conversation (temperature from settings)."""
    return _base_model(get_settings().llm_temperature)


def get_extract_model() -> ChatOpenAI:
    """Model for structured extraction (temperature 0 for determinism)."""
    return _base_model(0)


def get_bulk_model(max_tokens: int = 4096, timeout: int = 180) -> ChatOpenAI:
    """长输出客户端(离线脚本用:扩写知识文档、出题、挖事实)。

    业务侧 LLM_MAX_TOKENS 是 512,写一千字正文会被截断,而截断的表现往往是
    「请求超时」而不是报错——排查过一次就很难忘。所以脚本单独开一档。
    """
    settings = get_settings()
    kwargs = {}
    extra = _extra_body(settings)
    if extra:
        kwargs["extra_body"] = extra
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.4,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
        **kwargs,
    )


def get_judge_model() -> tuple[ChatOpenAI, str]:
    """评测用的裁判模型(ch04 评测真实化)。

    返回 (模型, 实际模型名)。judge_model 为空就回落生成模型——此时报告里必须标注
    「裁判=生成模型」,因为自己判自己的分数只能横向比策略,不能当准确率。

    裁判换厂商时 base_url / api_key 各自独立解析,只改名不换端点等于没换模型。
    """
    settings = get_settings()
    if not settings.judge_model:
        return _base_model(0), settings.llm_model
    kwargs = {}
    extra = _extra_body(settings)
    if extra:
        kwargs["extra_body"] = extra
    return (
        ChatOpenAI(
            model=settings.judge_model,
            base_url=settings.judge_base_url or settings.llm_base_url,
            api_key=settings.judge_api_key or settings.llm_api_key,
            temperature=0,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
            **kwargs,
        ),
        settings.judge_model,
    )
