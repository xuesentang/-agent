"""证据充分性自评单测(ch04):够/不够/判不了一律放行的取舍、入池理由。"""

from langchain_core.messages import AIMessage

from app.rag.selfcheck import ABSTAIN_REPLY, check_evidence


class FakeModel:
    def __init__(self, content: str, raises: Exception | None = None):
        self.content = content
        self.raises = raises
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.raises:
            raise self.raises
        return AIMessage(content=self.content)


async def test_empty_evidence_is_not_useful():
    result = await check_evidence("随便问问", "   ")
    assert result.useful is False
    assert "没有" in result.reason


async def test_model_says_useful_true():
    result = await check_evidence("运费怎么算", "证据…", model=FakeModel('{"useful": true, "reason": "有明确规则"}'))
    assert result.useful is True and result.degraded is False


async def test_model_says_useful_false_keeps_reason():
    model = FakeModel('{"useful": false, "reason": "证据只讲退货不讲积分"}')
    result = await check_evidence("积分怎么兑", "证据…", model=model)
    assert result.useful is False
    assert "积分" in result.reason


async def test_model_failure_fails_open():
    """判不了就当够:一次上游抖动不能让所有正常问题都被拒答。"""
    model = FakeModel("", raises=RuntimeError("500"))
    result = await check_evidence("运费怎么算", "证据…", model=model)
    assert result.useful is True and result.degraded is True


async def test_garbage_output_fails_open():
    result = await check_evidence("运费怎么算", "证据…", model=FakeModel("我觉得还行"))
    assert result.useful is True and result.degraded is True


def test_abstain_reply_does_not_promise_anything():
    """拒答话术:明确说没查到 + 给下一步,不编任何数字/时效。"""
    assert "没有找到可靠的依据" in ABSTAIN_REPLY
    assert "工单" in ABSTAIN_REPLY
    assert "工作日" not in ABSTAIN_REPLY
