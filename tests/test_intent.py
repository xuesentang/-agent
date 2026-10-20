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
    ROUTE_REFUND,
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


def test_eight_intents_map_to_five_routes():
    assert set(INTENT_ROUTES) == set(INTENTS)
    assert len(INTENTS) == 8  # 七类 + 「其他」兜底(CH06 第 4 点)
    assert set(INTENT_ROUTES.values()) == {
        ROUTE_KNOWLEDGE,
        ROUTE_BUSINESS,
        ROUTE_COMPLAINT,
        ROUTE_CHITCHAT,
        ROUTE_REFUND,
    }
    # 商品咨询走知识口(检索 → 闸 → Agent)
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_KNOWLEDGE] == ["商品咨询"]
    # 退款退货与售后合到退款子流程(CH06 第 5 点:先拿订单数据 + 强制检索政策)
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_REFUND] == ["退款退货", "售后"]
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_COMPLAINT] == ["投诉"]
    assert [k for k, v in INTENT_ROUTES.items() if v == ROUTE_CHITCHAT] == ["闲聊"]
    assert INTENT_ROUTES["其他"] == ROUTE_BUSINESS


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
    assert r.intent == FALLBACK_INTENT == "其他"
    assert r.route == ROUTE_BUSINESS
    assert r.fallback is True


def test_confidence_is_clamped():
    assert parse_intent('{"intent":"闲聊","confidence":5}').confidence == 1.0
    assert parse_intent('{"intent":"闲聊","confidence":-1}').confidence == 0.0
    assert parse_intent('{"intent":"闲聊","confidence":"高"}').confidence == 0.0


async def test_classify_uses_model_output():
    r = await classify_intent("我的鞋开胶了能退吗", model=FakeIntentModel('{"intent":"退款退货","confidence":0.88}'))
    assert r.intent == "退款退货"
    assert r.route == ROUTE_REFUND


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


async def test_history_accepts_langchain_messages():
    """真链路上 history 是 trim_history 产出的消息对象,不是 dict。

    这条是回归测试:第 6 章验收时线上报 'HumanMessage' object has no attribute 'get',
    因为这里只按 dict 取。单测一直喂 dict,所以没拦住。
    """
    from langchain_core.messages import HumanMessage

    model = FakeIntentModel('{"intent":"订单","confidence":0.9}')
    await classify_intent(
        "那个多少钱",
        model=model,
        history=[HumanMessage(content="订单 1001 买的是什么"), AIMessage(content="我帮您查一下")],
    )
    prompt = model.prompts[0]
    assert "最近对话" in prompt
    assert "订单 1001 买的是什么" in prompt
