"""三段式上下文:按两个锚点切历史、按角色截短中间层、按固定顺序拼给模型。

## 为什么是三层

一整通会话可能几十轮,全塞进上下文只有两种结局:爆窗口,或者每轮都花钱重新读一遍
早就无关的寒暄。一刀切「只留最近 N 条」又太狠:用户第三轮报的订单号、第五轮说的
「质量问题」,到第十五轮还得算数。

三段的分工:
- **层 1**(离当前最近):原文,一个字不压。要的就是「刚才说的还能原样看到」;
- **层 2**(中间):用户原话不动——诉求和单号都在里面;客服答复只留开头,答复的
  后半段多是解释和客套;工具结果换成一行标识,它本来就是给上一轮看的;
- **层 3**(最远):压成梗概,只留事实与诉求。

## 为什么不搬数据

降级只挪 `layer1_from_msg_id` 和 `summary_upto_msg_id` 两个 id,消息行永远不动。
把老消息改写或搬走,前端回载历史就看不到原样了(需求 5 要求切回旧会话能回载原文)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.token_budget import estimate_tokens

logger = logging.getLogger(__name__)

# 层 2 里客服答复保留的开头字数(需求 1:「只留开头几十个字」)
DEFAULT_ASSISTANT_HEAD = 40
# 工具结果在层 2 里被替换成的一行标识的前缀
TOOL_MARK = "[工具结果"
# 层 1 超预算时降级到「预算的这个比例」为止,而不是一条一条挪:
# 一次挪一条会让层 2 长期攒不起来,摘要永远等不到触发条件。
DEMOTE_TO_RATIO = 0.8


@dataclass
class HistoryItem:
    """一条历史消息 + 它在 messages 表里的 id。锚点比的是 id,所以必须带着走。"""

    msg_id: int
    message: BaseMessage


@dataclass
class LayerPlan:
    layer2: list[BaseMessage] = field(default_factory=list)
    layer1: list[BaseMessage] = field(default_factory=list)
    layer1_from_msg_id: int = 0  # 超预算时会前移,调用方据此更新锚点
    demoted: int = 0  # 层 1 被降级(挤到层 2)的条数
    tokens_layer1: int = 0
    tokens_layer2: int = 0
    # 截断前的层 2 用量:摘要的触发判据。只看截断后的值永远等于「刚好装得下」,
    # 触发器就没机会响——层 2 会一直悄悄丢消息。
    tokens_layer2_raw: int = 0
    dropped: int = 0  # 降级后仍装不下、被彻底丢掉的最早条数

    @property
    def window_size(self) -> int:
        return len(self.layer2) + len(self.layer1)


def _msg_id_of(item: HistoryItem) -> int:
    return int(item.msg_id or 0)


def _text_tokens(message: BaseMessage) -> int:
    tokens = estimate_tokens(getattr(message, "content", "") or "")
    for call in getattr(message, "tool_calls", None) or []:
        tokens += estimate_tokens(str(call.get("name", "")) + str(call.get("args", "")))
    return tokens


def render_layer2(message: BaseMessage, *, assistant_head: int = DEFAULT_ASSISTANT_HEAD) -> BaseMessage:
    """层 2 的单条截短。用户原话不动是硬规矩——判据都在用户说过的话里。"""
    if isinstance(message, HumanMessage):
        return message
    if isinstance(message, ToolMessage):
        size = len(str(message.content or ""))
        # 换成普通 ai 消息而不是保留 ToolMessage:层 2 的 assistant 已经被截成纯
        # 文本、没有 tool_calls 了,再挂一条 tool 角色就是孤儿,上游 400。
        return AIMessage(content=f"{TOOL_MARK} {size} 字,已省略]")
    if isinstance(message, AIMessage):
        content = str(message.content or "")
        if message.tool_calls and not content:
            names = ",".join(str(c.get("name")) for c in message.tool_calls)
            # 必须丢掉 tool_calls:层 2 只留一句标识,配对的 tool 结果可能被切到
            # 层 1 或已经被截成一行。留着 tool_calls 会让 sanitize_tool_pairs 误判
            # 「配对完整」而保留下来,上游照旧 400。
            return AIMessage(content=f"[发起工具调用 {names}]")
        if len(content) <= assistant_head:
            if message.tool_calls:
                return AIMessage(content=content)
            return message
        return AIMessage(content=content[:assistant_head] + "…")
    return message


def plan_layers(
    items: list[HistoryItem],
    *,
    summary_upto_msg_id: int,
    layer1_from_msg_id: int,
    layer1_budget: int,
    layer2_budget: int,
    assistant_head: int = DEFAULT_ASSISTANT_HEAD,
) -> LayerPlan:
    """按锚点切三层,并处理层 1 超预算的降级。

    降级规则(需求 2「层 1 超预算就降级一批」):从层 1 最早那端往外挪,挪进层 2
    (截短后进)。挪到装得下为止,新的起点回写给调用方去更新锚点。
    """
    plan = LayerPlan(layer1_from_msg_id=layer1_from_msg_id)
    if not items:
        return plan

    ordered = sorted(items, key=_msg_id_of)
    remaining = [it for it in ordered if _msg_id_of(it) > summary_upto_msg_id]
    layer1_items = [it for it in remaining if _msg_id_of(it) >= layer1_from_msg_id]
    layer2_items = [it for it in remaining if _msg_id_of(it) < layer1_from_msg_id]

    # 锚点为 0(新会话或从未降级)时全部消息都算层 1——装得下就不压
    tokens_layer1 = sum(_text_tokens(it.message) for it in layer1_items)
    # 降到 8 成以下才停:挪一条就够是「刚好不超」,层 2 会一直只攒一两条,
    # 层 2 的预算永远够,摘要永远不触发——级联断在这一环,后面两段都白写。
    # 但至少留一条:层 1 空掉等于把用户刚说的话丢了,比超预算更糟。
    target = max(int(layer1_budget * DEMOTE_TO_RATIO), 1)
    while len(layer1_items) > 1 and tokens_layer1 > target:
        moved = layer1_items.pop(0)
        tokens_layer1 -= _text_tokens(moved.message)
        layer2_items.append(moved)
        plan.demoted += 1

    plan.layer1 = [it.message for it in layer1_items]
    plan.tokens_layer1 = tokens_layer1
    # 新的层 1 起点 = 剩下层 1 的第一条;层 1 空了就取最后一条之后
    plan.layer1_from_msg_id = (
        _msg_id_of(layer1_items[0]) if layer1_items else (ordered[-1].msg_id + 1)
    )

    rendered = [render_layer2(it.message, assistant_head=assistant_head) for it in layer2_items]
    tokens_layer2 = sum(_text_tokens(m) for m in rendered)
    plan.tokens_layer2_raw = tokens_layer2
    # 层 2 也装不下:丢掉最早的一段(它已经进过摘要流程或被降级过,只能牺牲)
    while rendered and tokens_layer2 > layer2_budget:
        dropped = rendered.pop(0)
        tokens_layer2 -= _text_tokens(dropped)
        plan.dropped += 1
    plan.layer2 = rendered
    plan.tokens_layer2 = tokens_layer2
    return plan


def sanitize_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """去掉不成对的工具消息。

    上游按 Chat Completions 协议校验:`role=tool` 必须紧跟带对应 `tool_calls` 的
    assistant 消息。三层切分会把一对「助手发起工具调用 → 工具结果」切到不同层:
    层 2 里那条 assistant 被截成一句 `[发起工具调用 query_order]`(纯文本、没有
    tool_calls),层 1 开头就剩一条孤儿 tool 消息 —— 上游直接 400。

    这条在 CH07 之前不会出现,因为历史从不跨层。修法只有两步:配对不全的 assistant
    退回纯文本,孤儿 tool 消息丢掉。
    """
    out: list[BaseMessage] = []
    i, n = 0, len(messages)
    while i < n:
        msg = messages[i]
        calls = getattr(msg, "tool_calls", None) or []
        if isinstance(msg, AIMessage) and calls:
            ids = [c.get("id") for c in calls]
            j = i + 1
            following: list[ToolMessage] = []
            while j < n and isinstance(messages[j], ToolMessage):
                following.append(messages[j])
                j += 1
            matched = [t for t in following if t.tool_call_id in ids]
            if len(matched) == len(ids):
                out.append(msg)
                out.extend(following)
                i = j
                continue
            names = ",".join(str(c.get("name") or "?") for c in calls)
            text = str(msg.content or "") or f"[发起工具调用 {names}]"
            out.append(AIMessage(content=text))
            i = j if following else i + 1
            continue
        if isinstance(msg, ToolMessage):
            i += 1  # 前面没有对应的 tool_calls:孤儿,丢掉
            continue
        out.append(msg)
        i += 1
    return out


def assemble_messages(
    *,
    system_prompt: str,
    plan: LayerPlan,
    question: str,
    summary: str = "",
    evidence: str = "",
) -> list[BaseMessage]:
    """固定顺序:system → 层 2 → 层 1 → 当前这句 →(梗概 + 证据)。

    梗概和证据合成一条、挂在用户这句**之后**,不单独占 system:
    上游模板会把所有 system 上提合并渲染,工具定义就会被挤到可变内容之后,
    前缀缓存整段作废(需求 3)。放在最后还有个副作用是好的——它是每轮唯一变动的
    「背景块」,位置固定,前缀稳定性不受影响。
    """
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    messages.extend(plan.layer2)
    messages.extend(plan.layer1)
    messages.append(HumanMessage(content=question))
    background: list[str] = []
    if summary:
        background.append(f"【早前对话梗概】{summary}")
    if evidence:
        background.append(f"【知识库检索结果】以下是为你准备的参考资料,回答时按编号引用,资料没覆盖的不要编。\n{evidence}")
    if background:
        messages.append(HumanMessage(content="\n\n".join(background)))
    return sanitize_tool_pairs(messages)


def plan_from_history(
    history: list[BaseMessage] | None, *, layer2_count: int = 0
) -> LayerPlan:
    """服务层已经切好层,这里只是把它们装回一个 plan 好打日志。

    直接构造 `LayerPlan(layer1=...)` 会让 `tokens_layer1` 停在 0——日志里就会出现
    「layer1=33 条 / 0 token」这种自相矛盾的数,排查时最误导人。

    `layer2_count` 是服务层切完层之后报上来的条数:合并后的历史里前 n 条是层 2。
    不给就全算层 1——两者总条数一样,但 token 归属不同,日志里
    「layer1 超预算」这种假警报全来自这里。
    """
    items = list(history or [])
    split = max(0, min(int(layer2_count or 0), len(items)))
    layer2, layer1 = items[:split], items[split:]
    return LayerPlan(
        layer2=layer2,
        layer1=layer1,
        tokens_layer1=sum(_text_tokens(m) for m in layer1),
        tokens_layer2=sum(_text_tokens(m) for m in layer2),
    )


def render_model_ctx(plan: LayerPlan, *, summary: str, question: str) -> str:
    """主力 Agent 轮发给模型的东西,逐条打出来(需求 6)。"""
    lines = [
        f"model_ctx summary={len(summary)}字 window={plan.window_size}条 "
        f"(layer2={len(plan.layer2)}/{plan.tokens_layer2}t layer1={len(plan.layer1)}/{plan.tokens_layer1}t)"
    ]
    if summary:
        lines.append(f"  [梗概] {summary}")
    for msg, layer in [(m, "L2") for m in plan.layer2] + [(m, "L1") for m in plan.layer1]:
        role = getattr(msg, "type", "?")
        text = str(getattr(msg, "content", "") or "").replace("\n", " ")[:120]
        lines.append(f"  [{layer}]{role}: {text}")
    lines.append(f"  [当前] user: {question[:120]}")
    if plan.demoted or plan.dropped:
        lines.append(f"  降级:层1 降级 {plan.demoted} 条,层2 丢弃 {plan.dropped} 条")
    return "\n".join(lines)


def render_history_ctx(plan: LayerPlan, *, summary: str, question: str) -> str:
    """指代消解/意图共用的历史——闲聊兜底轮也要打(需求 6)。"""
    return (
        f"history_ctx summary={'有' if summary else '无'}({len(summary)}字) "
        f"window={plan.window_size}条\n"
        f"  [滑窗] " + " | ".join(
            f"{getattr(m, 'type', '?')}:{str(getattr(m, 'content', '') or '')[:40]}" for m in plan.layer2 + plan.layer1
        )
        + f"\n  [当前] {question[:80]}"
    )
