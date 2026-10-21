"""上下文预算:从模型窗口倒推能留给历史多少 token,再按 7:3 分给层 1 和层 2。

为什么不写死成一个常量:窗口、输出预留、工具结果上限、ReAct 步数都是变量,
写死的数字在默认配置下刚好、换一组配置就爆。第七章验收 2 就是拿一组演示配置
把这套算式跑一遍——如果预算是常量,那组配置下只会「看起来能用」然后半路 400。

**token 口径是这里最容易埋雷的地方。** LangChain 自带的
`count_tokens_approximately` 按英文 BPE 数中文:`build_agent_system_prompt`
1061 个汉字它报 5305——五倍。用它做预算等于把窗口砍掉五分之四,而且项目从
CH01 起就这么用了(配 2048 的预算实际只装得下约 400 token),只是历史一直不长
才没暴露。所以这里自己数:汉字与全角标点算 1 token,其余 4 字符 1 token,
向上取整——对中文偏保守,而预算这东西宁可少留一段历史,不能算少了等着上游拒。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 单块证据的估值:知识库块 150-400 字,按 300 token 估(含编号与章节路径)。
EVIDENCE_TOKENS_PER_CHUNK = 300
# ReAct 每步模型的正文输出估值(「我这就去查」+ 一段解释)。
PER_STEP_OUTPUT_TOKENS = 300
# 摘要段预留:需求要求梗概控制在几十到一两百字,取上限。
SUMMARY_RESERVED_TOKENS = 200
# 安全余量:窗口的 5%。上游的模板渲染、消息包装会额外吃一点,不留就卡在边界上。
SAFETY_RATIO = 0.05
# 层 1 占历史预算的比例,其余归层 2。
LAYER1_RATIO = 0.7
# 每个工具的 function 包装开销(`{"type":"function","function":{...}}` 那些键)。
# 不数它会让固定开销偏低几十 token,预算算出来「还装得下」,实际贴着边界。
TOOL_WRAPPER_TOKENS = 8


def _is_wide(ch: str) -> bool:
    """汉字 / 全角标点 / 全角字母 = 一个 token。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK 基本区
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x3000 <= code <= 0x303F  # 中文标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
        or 0xF900 <= code <= 0xFAFF
    )


def estimate_tokens(text: str | None) -> int:
    """粗估 token。中文按字、英文按 4 字符,向上取整。

    宁可高估:低估会让预算算出「还装得下」,然后上游 400。
    """
    if not text:
        return 0
    wide = sum(1 for ch in text if _is_wide(ch))
    rest = len(text) - wide
    return wide + math.ceil(rest / 4)


@dataclass(frozen=True)
class ContextBudget:
    """一轮对话的预算分配。所有字段单位都是 token。"""

    window: int
    reserved_system: int
    reserved_evidence: int
    reserved_summary: int
    reserved_output: int
    reserved_user_input: int
    reserved_react_peak: int
    safety: int
    history_total: int
    layer1: int
    layer2: int

    @property
    def fits(self) -> bool:
        return self.history_total > 0

    def as_log(self) -> str:
        return (
            f"window={self.window} history={self.history_total} "
            f"layer1={self.layer1} layer2={self.layer2} "
            f"system={self.reserved_system} evidence={self.reserved_evidence} "
            f"summary={self.reserved_summary} output={self.reserved_output} "
            f"user_input={self.reserved_user_input} react_peak={self.reserved_react_peak} "
            f"safety={self.safety}"
        )


def tool_schema_tokens(tools: list | None = None) -> int:
    """工具定义也要算进固定开销。

    需求第 3 点把「system 人设、红线和工具定义」算作每轮不变的那一段——它们一起
    构成前缀缓存的对象,不算进预算就是自欺:5 个内置工具约 350 token,占了小窗口
    配置里历史预算的 6%。
    """
    if tools is None:
        try:
            from app.core.tools.business_tools import build_default_registry

            tools = build_default_registry().for_bind()
        except Exception as exc:  # noqa: BLE001 - 数不出来就当 0,别把预算算崩
            logger.warning("工具定义 token 估算失败,按 0 计:%s", exc)
            return 0
    total = 0
    for t in tools:
        payload = {"name": getattr(t, "name", ""), "description": getattr(t, "description", ""), "parameters": getattr(t, "args", {})}
        total += estimate_tokens(json.dumps(payload, ensure_ascii=False)) + TOOL_WRAPPER_TOKENS
    return total


def compute_budget(
    *,
    window: int,
    system_tokens: int,
    tool_tokens: int = 0,
    max_output_tokens: int,
    max_user_input_tokens: int,
    max_agent_steps: int,
    tool_result_max_tokens: int,
    rerank_top_k: int,
    summary_tokens: int = SUMMARY_RESERVED_TOKENS,
) -> ContextBudget:
    """从窗口倒推历史预算。固定开销先全扣,剩下的才是历史。"""
    evidence = max(rerank_top_k, 0) * EVIDENCE_TOKENS_PER_CHUNK
    react_peak = max(max_agent_steps, 1) * (tool_result_max_tokens + PER_STEP_OUTPUT_TOKENS)
    safety = int(window * SAFETY_RATIO)
    fixed_system = system_tokens + tool_tokens
    fixed = fixed_system + evidence + summary_tokens + max_output_tokens + max_user_input_tokens
    history = window - fixed - react_peak - safety
    if history <= 0:
        logger.warning(
            "上下文预算不足:window=%s 固定开销=%s(system %s + tools %s + evidence %s + summary %s "
            "+ output %s + user_input %s)+ react_peak=%s + safety=%s,历史预算 %s",
            window,
            fixed,
            system_tokens,
            tool_tokens,
            evidence,
            summary_tokens,
            max_output_tokens,
            max_user_input_tokens,
            react_peak,
            safety,
            history,
        )
        history = 0
    layer1 = int(history * LAYER1_RATIO)
    return ContextBudget(
        window=window,
        reserved_system=fixed_system,
        reserved_evidence=evidence,
        reserved_summary=summary_tokens,
        reserved_output=max_output_tokens,
        reserved_user_input=max_user_input_tokens,
        reserved_react_peak=react_peak,
        safety=safety,
        history_total=history,
        layer1=layer1,
        layer2=history - layer1,
    )


def budget_from_settings(system_prompt: str):
    """按 .env 配置 + 实测 system prompt/工具定义算一遍。启动自检和每轮都用它。"""
    from app.config import get_settings

    s = get_settings()
    return compute_budget(
        window=s.model_context_window,
        system_tokens=estimate_tokens(system_prompt),
        tool_tokens=tool_schema_tokens(),
        max_output_tokens=s.max_output_tokens,
        max_user_input_tokens=s.max_user_input_tokens,
        max_agent_steps=s.agent_max_steps,
        tool_result_max_tokens=s.tool_result_max_tokens,
        rerank_top_k=s.rerank_top_k,
    )


def log_startup_check() -> ContextBudget:
    """启动自检:打一行账单;连一轮都装不下就明确报警(需求 4)。"""
    from app.core.prompts import build_agent_system_prompt
    from app.config import get_settings

    budget = budget_from_settings(build_agent_system_prompt(max_steps=get_settings().agent_max_steps))
    if budget.fits:
        logger.info("context.budget %s", budget.as_log())
    else:
        logger.warning(
            "上下文预算不足:窗口 %s 连固定开销都装不下,历史将为 0 条(每轮只带当前这句话)",
            budget.window,
        )
    return budget
