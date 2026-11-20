"""Smoke tests for the eligibility-claims web UI.

The UI drives the real Strands agent on the scripted mock model,
through the real MCP server — so these assert the same verdicts the
CLI demo prints. If a marker disappears, the wiring broke. Run from
the repo root:

    python3 -m pytest web -q
"""

import os
import uuid

# Mock mode only: never let a stray key (or a local .env) turn the
# test run live. The agent is built lazily on the first question.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from web.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_loads_with_chat_panel_and_mock_data():
    response = client.get("/")
    assert response.status_code == 200
    # Pain framing + honest scope, up top.
    assert "20 minutes on hold" in response.text
    assert "Demo on synthetic mock data" in response.text
    # The chat panel: placeholder + sample questions.
    assert "Is member MBR-1001 covered" in response.text
    assert "Eligibility — covered" in response.text
    assert "Submit a claim (asks first)" in response.text
    # Mock data reference: members, procedures, seed claims.
    assert "MBR-1001" in response.text
    assert "70553" in response.text
    assert "CLM-9003" in response.text
    # Mock mode is the default.
    assert "scripted mock mode" in response.text


def test_system_prompt_shown():
    response = client.get("/")
    assert response.status_code == 200
    assert "System prompt" in response.text
    # The agent's system prompt, verbatim from agent.py.
    assert "You are a healthcare eligibility and claims assistant" in response.text


def test_eligibility_question_answers_with_tool_call():
    response = client.post(
        "/chat",
        data={"message": "Check eligibility for member MBR-1001 for procedure 99213."},
    )
    assert response.status_code == 200
    assert "is eligible" in response.text
    # The tool call is shown with its inputs and result.
    assert "verify_eligibility" in response.text
    assert "Acme Health PPO Gold" in response.text


def test_terminated_member_is_not_eligible():
    response = client.post(
        "/chat",
        data={"message": "Check eligibility for member MBR-1003 for procedure 70553."},
    )
    assert response.status_code == 200
    assert "NOT eligible" in response.text
    assert "terminated" in response.text


def test_claim_status_denied():
    response = client.post(
        "/chat", data={"message": "What is the status of claim CLM-9003?"}
    )
    assert response.status_code == 200
    assert "check_claim_status" in response.text
    assert "denied" in response.text
    assert "Prior authorization" in response.text


def test_cost_estimate():
    response = client.post(
        "/chat",
        data={
            "message": "Estimate the patient cost for member MBR-1002, "
            "procedure 70553, billed $2400.00."
        },
    )
    assert response.status_code == 200
    assert "estimate_patient_cost" in response.text
    assert "1344" in response.text  # same number as `make demo-mock`


def test_submit_without_confirmation_shows_confirm_panel():
    key = f"web-test-{uuid.uuid4().hex[:8]}"
    response = client.post(
        "/chat",
        data={
            "message": "Submit a claim for member MBR-1001, procedure 99213, "
            f"billed $250.00, provider NPI-1111111111, idempotency key {key}."
        },
    )
    assert response.status_code == 200
    # The tool refused; the UI asks the human to confirm. No claim yet.
    assert "Confirm before submitting" in response.text
    assert "Yes — submit this claim" in response.text
    assert "Claim submitted:" not in response.text


def test_confirm_flow_submits_once_then_idempotent():
    key = f"web-test-{uuid.uuid4().hex[:8]}"
    form = {
        "member_id": "MBR-1001",
        "procedure_code": "99213",
        "billed_amount": "250.0",
        "provider_npi": "NPI-1111111111",
        "idempotency_key": key,
    }
    first = client.post("/confirm", data=form)
    assert first.status_code == 200
    assert "Claim submitted:" in first.text

    second = client.post("/confirm", data=form)
    assert second.status_code == 200
    # Same key again: the original claim comes back, no duplicate.
    assert "Already submitted" in second.text


def test_empty_message_is_a_friendly_error():
    response = client.post("/chat", data={"message": "  "})
    assert response.status_code == 200
    assert "Type a question first" in response.text


def test_unknown_member_fails_gracefully():
    response = client.post(
        "/chat",
        data={"message": "Check eligibility for member MBR-9999 for procedure 99213."},
    )
    assert response.status_code == 200
    assert "not found" in response.text.lower()
