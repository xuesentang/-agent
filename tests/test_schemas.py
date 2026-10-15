import pytest
from pydantic import ValidationError

from app.core.schemas import AfterSalesInfo


def test_full_extraction():
    info = AfterSalesInfo(
        order_id="20241015001", claim_type="refund", expected_solution="全额退款"
    )
    assert info.claim_type == "refund"


def test_all_fields_optional():
    info = AfterSalesInfo()
    assert info.order_id is None
    assert info.claim_type is None
    assert info.expected_solution is None


def test_invalid_claim_type_rejected():
    with pytest.raises(ValidationError):
        AfterSalesInfo(claim_type="magic")


def test_field_descriptions_present():
    schema = AfterSalesInfo.model_json_schema()
    for field in ("order_id", "claim_type", "expected_solution"):
        assert schema["properties"][field].get("description")


def test_all_claim_types_accepted():
    for ct in ("refund", "return_refund", "exchange", "repair",
               "compensation", "complaint", "inquiry", "other"):
        assert AfterSalesInfo(claim_type=ct).claim_type == ct
