"""Unit tests for the deterministic logic — no MCP, no model, no network."""

import pytest

from agent_eligibility_claims import logic


@pytest.fixture(autouse=True)
def _fresh_store():
    logic.STORE.reset()
    yield
    logic.STORE.reset()


def test_eligibility_pass_deductible_met():
    out = logic.verify_eligibility("MBR-1001", "99213")
    assert out["eligible"] is True
    assert out["coverage_status"] == "active"
    assert out["in_network"] is True
    assert out["deductible_remaining"] == 0
    assert out["copay_office_visit"] == 25
    assert out["prior_auth_required"] is False


def test_eligibility_pass_deductible_remaining_and_prior_auth():
    out = logic.verify_eligibility("MBR-1002", "70553")
    assert out["eligible"] is True
    assert out["deductible_remaining"] == 1200
    assert out["prior_auth_required"] is True


def test_eligibility_terminated_coverage():
    out = logic.verify_eligibility("MBR-1003", "70553")
    assert out["eligible"] is False
    assert out["coverage_status"] == "terminated"
    assert "terminated" in out["reason"].lower()


def test_eligibility_out_of_network():
    # MBR-1004 is on an HMO whose only in-network provider is NPI-3333333333.
    out = logic.verify_eligibility("MBR-1004", "99213", "NPI-1111111111")
    assert out["eligible"] is False
    assert out["in_network"] is False
    assert "out-of-network" in out["reason"]


def test_eligibility_unknown_member_and_procedure():
    assert logic.verify_eligibility("MBR-9999", "99213")["eligible"] is False
    assert logic.verify_eligibility("MBR-1001", "00000")["eligible"] is False


def test_claim_status_paid_pending_denied_in_review():
    paid = logic.check_claim_status("CLM-9001")
    assert paid["found"] is True
    assert paid["status"] == "paid"
    assert paid["patient_responsibility"] == 200.0

    pending = logic.check_claim_status("CLM-9002")
    assert pending["status"] == "pending"

    denied = logic.check_claim_status("CLM-9003")
    assert denied["status"] == "denied"
    assert "prior authorization" in denied["denial_reason"].lower()

    assert logic.check_claim_status("CLM-9004")["status"] == "in_review"
    assert logic.check_claim_status("CLM-9999")["found"] is False


def test_cost_estimate_deductible_then_coinsurance():
    # MBR-1002: $1,200 deductible remaining, 20% coinsurance.
    # Billed $2,400 -> allowed $1,920 -> deductible $1,200 + 20% of $720 = $144.
    out = logic.estimate_patient_cost("MBR-1002", "70553", 2400.0)
    assert out["estimable"] is True
    assert out["allowed_amount"] == 1920.0
    assert out["deductible_applied"] == 1200.0
    assert out["coinsurance_owed"] == 144.0
    assert out["copay"] == 0.0
    assert out["patient_total"] == 1344.0
    assert out["plan_pays"] == 576.0
    assert out["provider_write_off"] == 480.0


def test_cost_estimate_deductible_met_office_visit_copay():
    # MBR-1001: deductible met, 20% coinsurance, $25 office-visit copay.
    # Billed $250 -> allowed $200 -> coinsurance $40 + copay $25 = $65.
    out = logic.estimate_patient_cost("MBR-1001", "99213", 250.0)
    assert out["patient_total"] == 65.0
    assert out["deductible_applied"] == 0.0
    assert out["coinsurance_owed"] == 40.0
    assert out["copay"] == 25.0


def test_cost_estimate_not_estimable_when_terminated():
    out = logic.estimate_patient_cost("MBR-1003", "70553", 2400.0)
    assert out["estimable"] is False


def test_submit_requires_confirmation():
    out = logic.submit_claim(
        "MBR-1001", "99213", 250.0, "NPI-1111111111", "key-no-confirm"
    )
    assert out["submitted"] is False
    assert "onfirm" in out["reason"]
    assert "key-no-confirm" not in logic.STORE.idempotency_keys


def test_submit_creates_claim_and_is_idempotent():
    first = logic.submit_claim(
        "MBR-1001", "99213", 250.0, "NPI-1111111111", "key-abc", confirmed=True
    )
    assert first["submitted"] is True
    assert first["duplicate"] is False
    assert first["claim_id"].startswith("CLM-91")

    second = logic.submit_claim(
        "MBR-1001", "99213", 250.0, "NPI-1111111111", "key-abc", confirmed=True
    )
    assert second["submitted"] is True
    assert second["duplicate"] is True
    assert second["claim_id"] == first["claim_id"]
    # Only one claim was created for the two calls.
    assert len(logic.STORE.claims) == 5  # 4 seed claims + 1 new


def test_submit_refused_for_terminated_member():
    out = logic.submit_claim(
        "MBR-1003", "70553", 2400.0, "NPI-1111111111", "key-term", confirmed=True
    )
    assert out["submitted"] is False
    assert "terminated" in out["reason"].lower()
