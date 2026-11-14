"""Orchestration tests: the Strands agent + scripted model + MCP tools."""

import re
import uuid

import pytest
from strands import Agent

from agent_eligibility_claims.agent import EXPECTED_TOOLS, SYSTEM_PROMPT, list_loaded_tools
from agent_eligibility_claims.scripted_model import ScriptedModel
from conftest import result_text


@pytest.fixture
def agent(mcp_client):
    a = Agent(
        model=ScriptedModel(),
        tools=[mcp_client],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    yield a
    # Stop the shared client so the next test's agent starts a fresh server.
    # stop() resets the client state for reuse.
    try:
        mcp_client.stop(None, None, None)
    except Exception:
        pass


def test_agent_loads_exactly_the_four_mcp_tools(agent, mcp_client):
    tools = list_loaded_tools(mcp_client)
    assert sorted(name for name, _ in tools) == sorted(EXPECTED_TOOLS)
    assert len(tools) == 4


def test_eligibility_check_that_passes(agent):
    text = result_text(agent("Check eligibility for member MBR-1001 for procedure 99213."))
    assert "eligible" in text.lower()
    assert "MBR-1001" in text
    assert "$0" in text or "0 deductible" in text


def test_eligibility_check_that_fails_terminated(agent):
    text = result_text(agent("Check eligibility for member MBR-1003 for procedure 70553."))
    assert "not eligible" in text.lower()
    assert "terminated" in text.lower()


def test_claim_status_lookup_denied(agent):
    text = result_text(agent("What is the status of claim CLM-9003?"))
    assert "denied" in text.lower()
    assert "prior authorization" in text.lower()


def test_cost_estimate(agent):
    text = result_text(
        agent(
            "Estimate the patient cost for member MBR-1002, procedure 70553, "
            "billed $2400.00."
        )
    )
    assert "1344" in text


def test_idempotent_submit_through_agent(agent):
    key = f"test-orch-{uuid.uuid4().hex[:8]}"
    prompt = (
        "Submit a claim for member MBR-1001, procedure 99213, billed $250.00, "
        f"provider NPI-1111111111, idempotency key {key}. I confirm this submission."
    )
    first = result_text(agent(prompt))
    assert "submitted" in first.lower()
    claim_id = re.search(r"CLM-\d+", first)
    assert claim_id, f"no claim id in first submit response: {first!r}"

    second = result_text(agent(prompt))
    assert "already submitted" in second.lower()
    assert claim_id.group(0) in second


def test_submit_without_confirmation_is_refused(agent):
    text = result_text(
        agent(
            "Submit a claim for member MBR-1001, procedure 99213, billed $250.00, "
            f"provider NPI-1111111111, idempotency key orch-noconsent-{uuid.uuid4().hex[:8]}."
        )
    )
    assert "didn't submit" in text.lower() or "confirmation" in text.lower()
