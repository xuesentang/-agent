"""意图识别单测:解析兜底 + 分流表 —— 模型部分靠标注样例 eval 验,这里只验确定性的那半。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.intent import (
    FALLBACK_INTENT,
    INTENT_ROUTES,
    INTENTS,
    ROUTE_BUSINESS,
    ROUTE_CHITCHAT,
    ROUTE_COMPLAINT,
    ROUTE_KNOWLEDGE,
    classify_intent,
    parse_intent,
    route_of,
)


class FakeIntentModel:
    def __init__(self, content: str = "", error: Exception | None = None, reject_json_mode=False):
        self._content = content
        self._error = error
        self._reject = reject_json_mode
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kwargs):
        self.prompts.append(str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1]))
        if self._reject and kwargs.get("response_format"):
            raise TypeError("response_format not supported")
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content)


def test_seven_intents_map_to_four_routes():
    assert set(INTENT_ROUTES) == set(INTENTS)
    assert len(INTENTS) == 7
    assert set(INTENT_ROUTES.values()) == {
        ROUTE_KNOWLEDGE,
        ROUTE_BUSINESS,
        ROUTE_COMPLAINT,
        ROUTE_CHITCHAT,
    }
    # 知识类只放这两个:它们才需要「先检索政策再说」
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_KNOWLEDGE] == ["商品咨询", "退款退货"]
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_COMPLAINT] == ["投诉"]
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_CHITCHAT] == ["闲聊"]


def test_unknown_intent_falls_back_to_business():
    assert route_of("外星语") == ROUTE_BUSINESS
    assert route_of(FALLBACK_INTENT) == ROUTE_BUSINESS


def test_parse_plain_json():
    r = parse_intent('{"intent": "投诉", "confidence": 0.92}')
    assert (r.intent, r.route, r.confidence, r.fallback) == ("投诉", ROUTE_COMPLAINT, 0.92, False)


def test_parse_json_wrapped_in_prose_and_fence():
    r = parse_intent('好的,判断如下:\n```json\n{"intent":"物流","confidence":0.8}\n```')
    assert r.intent == "物流"
    assert r.route == ROUTE_BUSINESS


def test_partial_intent_word_is_matched():
    """模型偶尔给「退款」这种半截词,不该因此判兜底。"""
    assert parse_intent('{"intent":"退款","confidence":0.7}').intent == "退款退货"


@pytest.mark.parametrize("bad", ["", "不是 JSON", '{"intent":"","confidence":0.5}', '{"foo":1}'])
def test_unparsable_falls_back_without_raising(bad):
    r = parse_intent(bad)
    assert r.intent == FALLBACK_INTENT
    assert r.route == ROUTE_BUSINESS
    assert r.fallback is True


def test_confidence_is_clamped():
    assert parse_intent('{"intent":"闲聊","confidence":5}').confidence == 1.0
    assert parse_intent('{"intent":"闲聊","confidence":-1}').confidence == 0.0
    assert parse_intent('{"intent":"闲聊","confidence":"高"}').confidence == 0.0


async def test_classify_uses_model_output():
    r = await classify_intent("我的鞋开胶了能退吗", model=FakeIntentModel('{"intent":"退款退货","confidence":0.88}'))
    assert r.intent == "退款退货"
    assert r.route == ROUTE_KNOWLEDGE


async def test_classify_survives_model_error():
    r = await classify_intent("随便", model=FakeIntentModel(error=RuntimeError("upstream 500")))
    assert r.fallback is True
    assert r.route == ROUTE_BUSINESS


async def test_classify_retries_without_json_mode():
    model = FakeIntentModel('{"intent":"订单","confidence":0.6}', reject_json_mode=True)
    r = await classify_intent("我的订单多少钱", model=model)
    assert r.intent == "订单"
    assert len(model.prompts) == 2  # 第一次被拒,退回普通调用


async def test_history_is_attached_only_when_given():
    model = FakeIntentModel('{"intent":"闲聊","confidence":0.5}')
    await classify_intent("那它呢", model=model, history=[{"role": "user", "content": "你们几点下班"}])
    assert "最近对话" in model.prompts[0]
    model2 = FakeIntentModel('{"intent":"闲聊","confidence":0.5}')
    await classify_intent("那它呢", model=model2)
    assert "最近对话" not in model2.prompts[0]
