"""理解层单测:指代补全、问法归一、槽位抽取、失败回落。

模型部分用假模型,验的是「协议怎么解析、失败怎么退」;真实补全质量靠
`tests/eval/ch06_dialogue_cases.jsonl` 的多轮 eval。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.query_understanding import expand_queries, guess_order_id, understand_query


class FakeModel:
    def __init__(self, content: str = "", error: Exception | None = None, reject_json=False):
        self._content = content
        self._error = error
        self._reject = reject_json
        self.calls = 0
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1]))
        if self._reject and kwargs.get("response_format"):
            raise TypeError("no response_format")
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content)


async def test_pronoun_is_resolved_with_history():
    model = FakeModel(
        '{"resolved": "订单 1001 的无线蓝牙降噪耳机能退吗", "canonical": "耳机 退货条件", "order_id": "1001", "changed": true}'
    )
    history = [
        {"role": "user", "content": "订单 1001 买了个耳机"},
        {"role": "assistant", "content": "已查到订单 1001"},
    ]
    r = await understand_query("它能退吗", history=history, model=model)
    assert r.resolved.startswith("订单 1001")
    assert r.canonical == "耳机 退货条件"
    assert r.order_id == "1001"
    assert r.changed is True
    assert "订单 1001 买了个耳机" in model.prompts[0]


async def test_summary_is_given_to_understand_early_orders():
    """早期对话被压成梗概后,「最开始那个订单」只能靠梗概指认。"""
    model = FakeModel('{"resolved": "订单 1001 的物流到哪了", "canonical": "物流进度", "order_id": "1001", "changed": true}')
    r = await understand_query(
        "最开始那个订单后来怎么说",
        history=[{"role": "user", "content": "订单 1007 到哪了"}],
        summary="订单 1001 待发货,用户问过物流;订单 1007 已签收,用户要退货。",
        model=model,
    )
    assert r.order_id == "1001"
    prompt = model.prompts[0]
    assert "订单 1001 待发货" in prompt, "梗概必须进提示词,否则时序指代无从判断"
    assert "时序指代" in prompt


async def test_complete_question_passes_through_unchanged():
    model = FakeModel('{"resolved": "邮费是多少", "canonical": "运费标准", "order_id": null, "changed": false}')
    r = await understand_query("邮费是多少", model=model)
    assert r.resolved == "邮费是多少"
    assert r.changed is False
    assert r.order_id is None


async def test_order_id_falls_back_to_regex_when_model_omits_it():
    """模型抽不到槽位时不能让订单号丢了——补全句子里已经写着了。"""
    model = FakeModel('{"resolved": "订单 1001 的耳机能退吗", "canonical": "耳机 退货条件"}')
    r = await understand_query("它能退吗", model=model)
    assert r.order_id == "1001"


async def test_model_error_passes_through():
    model = FakeModel(error=RuntimeError("upstream 500"))
    r = await understand_query("它能退吗", history=[{"role": "user", "content": "订单 2048"}], model=model)
    assert r.resolved == "它能退吗"
    assert r.fallback is True


async def test_garbage_json_passes_through():
    model = FakeModel("我理解你的意思是退款")
    r = await understand_query("它能退吗", model=model)
    assert r.resolved == "它能退吗"
    assert r.fallback is True


async def test_retries_without_json_mode():
    model = FakeModel('{"resolved": "a", "canonical": "b"}', reject_json=True)
    r = await understand_query("x", model=model)
    assert r.resolved == "a"
    assert model.calls == 2


def test_guess_order_id_ignores_short_numbers():
    assert guess_order_id("订单 1001 到哪了") == "1001"
    assert guess_order_id("3 天能到吗") is None
    assert guess_order_id("") is None


async def test_expand_returns_original_first_and_dedupes():
    model = FakeModel('{"queries": ["运费标准", "退货运费谁出", "运费标准"]}')
    qs = await expand_queries("退货运费怎么算", model=model)
    assert qs[0] == "退货运费怎么算"
    assert len(qs) == len(set(qs))
    assert "运费标准" in qs


async def test_expand_caps_at_three():
    model = FakeModel('{"queries": ["a", "b", "c", "d", "e"]}')
    qs = await expand_queries("q", model=model)
    assert len(qs) == 3


@pytest.mark.parametrize("bad", ["", "不是 json", '{"queries": "不是数组"}', '{"other": []}'])
async def test_expand_failure_falls_back_to_original(bad):
    qs = await expand_queries("退货运费怎么算", model=FakeModel(bad))
    assert qs == ["退货运费怎么算"]
