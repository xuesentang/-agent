"""三层切分与拼装单测:锚点边界、按角色截短、顺序、降级。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.context import (
    HistoryItem,
    assemble_messages,
    plan_layers,
    render_history_ctx,
    render_layer2,
    render_model_ctx,
)


def _items(n: int, *, size: int = 10) -> list[HistoryItem]:
    """造 n 条历史,id 从 100 起(避开 0 这个「未设置」哨兵)。"""
    out = []
    for i in range(n):
        msg = HumanMessage(content=f"问题{i}" + "补" * size) if i % 2 == 0 else AIMessage(content=f"答复{i}" + "答" * size)
        out.append(HistoryItem(msg_id=100 + i, message=msg))
    return out


def test_no_anchors_means_everything_is_layer1():
    plan = plan_layers(
        _items(4), summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=10_000, layer2_budget=10_000
    )
    assert len(plan.layer1) == 4
    assert plan.layer2 == []
    assert plan.demoted == 0 and plan.dropped == 0


def test_anchors_split_three_layers():
    items = _items(6)  # id 100..105
    plan = plan_layers(
        items,
        summary_upto_msg_id=101,
        layer1_from_msg_id=104,
        layer1_budget=10_000,
        layer2_budget=10_000,
    )
    assert [m.content[:3] for m in plan.layer1] == ["问题4", "答复5"]  # 104,105
    assert len(plan.layer2) == 2  # 102,103
    assert plan.tokens_layer1 > 0


def test_summary_covered_messages_are_not_sent():
    plan = plan_layers(
        _items(4),
        summary_upto_msg_id=101,
        layer1_from_msg_id=102,
        layer1_budget=10_000,
        layer2_budget=10_000,
    )
    sent = [m.content[:3] for m in plan.layer2 + plan.layer1]
    assert sent == ["问题2", "答复3"]


def test_layer1_over_budget_demotes_earliest():
    items = _items(6, size=100)  # 每条约 110 token
    plan = plan_layers(
        items, summary_upto_msg_id=0, layer1_from_msg_id=100, layer1_budget=250, layer2_budget=10_000
    )
    assert plan.demoted >= 4  # 250 token 装不下 6 条
    assert plan.window_size == 6
    # 被降级的进层 2,且第一个剩下的层 1 的 id 就是新锚点
    assert plan.layer1_from_msg_id == 100 + plan.demoted


def test_layer2_over_budget_drops_earliest():
    items = _items(6, size=100)
    # 层 1 只留最后一条,其余被降级进层 2,但层 2 预算极小 → 丢
    plan = plan_layers(
        items, summary_upto_msg_id=0, layer1_from_msg_id=100, layer1_budget=110, layer2_budget=1
    )
    assert plan.dropped >= 4
    assert plan.layer2 == []
    assert len(plan.layer1) == 1


def test_ai_answer_truncated_but_user_words_untouched():
    long_user = HumanMessage(content="我要退货" + "原因" * 100)
    long_ai = AIMessage(content="好的,我帮您查一下" + "说明" * 100)
    assert render_layer2(long_user).content == long_user.content
    truncated = render_layer2(long_ai, assistant_head=10).content
    assert truncated.startswith("好的,我帮您查一下"[:10]) and truncated.endswith("…")


def test_tool_result_becomes_one_line():
    msg = ToolMessage(content="物流轨迹:" + "x" * 500, tool_call_id="c1")
    rendered = render_layer2(msg)
    assert rendered.content.startswith("[工具结果") and len(rendered.content) < 30
    # 层 2 里不留 tool 角色:配对的 assistant 已被截成纯文本,留 tool 就是孤儿
    assert rendered.type == "ai"


def test_tool_call_message_without_text_is_marked():
    msg = AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1", "type": "tool_call"}])
    assert "query_order" in render_layer2(msg).content


def test_assemble_order_matches_requirement():
    plan = plan_layers(
        _items(4), summary_upto_msg_id=100, layer1_from_msg_id=102, layer1_budget=10_000, layer2_budget=10_000
    )
    messages = assemble_messages(
        system_prompt="SYS", plan=plan, question="当前这句", summary="早前提过订单 1001", evidence="[1] 政策"
    )
    assert isinstance(messages[0], SystemMessage)
    roles = [m.type for m in messages]
    # system →(层2:101 是答复)→(层1:102、103)→ 当前这句 → 背景块
    assert roles == ["system", "ai", "human", "ai", "human", "human"]
    assert messages[1].content.startswith("答复1")  # 层 2 里被截短的答复
    assert messages[-2].content == "当前这句"
    assert messages[-1].type == "human"
    assert "梗概" in messages[-1].content and "[1] 政策" in messages[-1].content


def test_background_block_is_skipped_when_empty():
    plan = plan_layers([], summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=1, layer2_budget=1)
    messages = assemble_messages(system_prompt="SYS", plan=plan, question="只有一句")
    assert len(messages) == 2
    assert messages[-1].content == "只有一句"


def test_ctx_logs_expose_summary_and_window():
    plan = plan_layers(
        _items(3), summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=10_000, layer2_budget=10_000
    )
    model_ctx = render_model_ctx(plan, summary="用户报过订单 1001", question="它能退吗")
    assert "model_ctx" in model_ctx and "用户报过订单 1001" in model_ctx
    history_ctx = render_history_ctx(plan, summary="", question="它能退吗")
    assert "history_ctx" in history_ctx and "window=3条" in history_ctx


def test_plan_from_history_splits_layers_by_count():
    """服务层报上来的条数决定 token 归属:不然日志里层 1 会显示成超预算。"""
    from app.core.context import plan_from_history

    history = [
        HumanMessage(content="很久以前的问题" + "旧" * 200),
        AIMessage(content="很久以前的答复" + "答" * 200),
        HumanMessage(content="刚才那句"),
    ]
    plan = plan_from_history(history, layer2_count=2)
    assert len(plan.layer2) == 2 and len(plan.layer1) == 1
    assert plan.tokens_layer2 > plan.tokens_layer1
    assert plan.window_size == 3
    # 不报条数就全算层 1(老行为,兼容没有分层信息的调用方)
    legacy = plan_from_history(history)
    assert legacy.layer2 == [] and legacy.tokens_layer1 > 0


def test_orphan_tool_message_is_dropped():
    """跨层后层 1 开头可能是孤儿 tool 消息,上游会 400。"""
    from app.core.context import sanitize_tool_pairs

    msgs = [
        SystemMessage(content="SYS"),
        ToolMessage(content="订单 1001 已发货", tool_call_id="c1"),
        HumanMessage(content="那它能退吗"),
    ]
    out = sanitize_tool_pairs(msgs)
    assert [m.type for m in out] == ["system", "human"]


def test_unpaired_tool_call_message_degrades_to_text():
    from app.core.context import sanitize_tool_pairs

    ai = AIMessage(
        content="",
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}],
    )
    out = sanitize_tool_pairs([SystemMessage(content="SYS"), ai])
    assert out[-1].type == "ai"
    assert out[-1].tool_calls == []
    assert "query_order" in out[-1].content


def test_paired_tool_messages_are_kept_intact():
    from app.core.context import sanitize_tool_pairs

    ai = AIMessage(
        content="我查一下",
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1", "type": "tool_call"}],
    )
    tool = ToolMessage(content="已发货", tool_call_id="c1")
    out = sanitize_tool_pairs([SystemMessage(content="SYS"), ai, tool])
    assert [m.type for m in out] == ["system", "ai", "tool"]
    assert out[1].tool_calls[0]["id"] == "c1"


def test_assemble_never_emits_tool_without_tool_calls():
    """层 2 里那条 assistant 被截成纯文本后,配对的 tool 必须一起消失。"""
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "query_order", "args": {}, "id": "c1", "type": "tool_call"}],
    )
    tool = ToolMessage(content="x" * 300, tool_call_id="c1")
    items = [
        HistoryItem(msg_id=100, message=ai),
        HistoryItem(msg_id=101, message=tool),
        HistoryItem(msg_id=102, message=HumanMessage(content="那它能退吗")),
    ]
    plan = plan_layers(
        items, summary_upto_msg_id=0, layer1_from_msg_id=102, layer1_budget=10_000, layer2_budget=10_000
    )
    messages = assemble_messages(system_prompt="SYS", plan=plan, question="当前")
    types = [m.type for m in messages]
    assert "tool" not in types
    for i, m in enumerate(messages):
        if m.type == "ai" and getattr(m, "tool_calls", None):
            assert i + 1 < len(messages) and messages[i + 1].type == "tool"
