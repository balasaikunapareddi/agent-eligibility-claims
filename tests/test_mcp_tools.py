"""Tests that call the MCP tools over the real protocol (stdio client)."""

import json
import uuid

from agent_eligibility_claims.agent import EXPECTED_TOOLS, list_loaded_tools


def call_tool(client, name, args):
    """Call one MCP tool and return its JSON result as a dict."""
    result = client.call_tool_sync(f"test-{uuid.uuid4().hex[:8]}", name, args)
    if isinstance(result, dict):
        structured = result.get("structuredContent")
        if structured:
            return structured
        blocks = result.get("content", [])
    else:
        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        blocks = getattr(result, "content", [])
    texts = []
    for block in blocks:
        if isinstance(block, dict):
            texts.append(block.get("text", ""))
        else:
            texts.append(getattr(block, "text", ""))
    return json.loads("".join(texts))


def test_agent_loads_exactly_four_mcp_tools(mcp_client_started):
    tools = list_loaded_tools(mcp_client_started)
    names = sorted(name for name, _ in tools)
    assert names == sorted(EXPECTED_TOOLS)
    assert len(tools) == 4
    # Every tool carries a one-line description for the startup listing.
    for name, description in tools:
        assert description, f"{name} has no description"


def test_mcp_verify_eligibility(mcp_client_started):
    out = call_tool(
        mcp_client_started,
        "verify_eligibility",
        {"member_id": "MBR-1001", "procedure_code": "99213"},
    )
    assert out["eligible"] is True
    assert out["deductible_remaining"] == 0

    out = call_tool(
        mcp_client_started,
        "verify_eligibility",
        {"member_id": "MBR-1003", "procedure_code": "70553"},
    )
    assert out["eligible"] is False
    assert out["coverage_status"] == "terminated"


def test_mcp_check_claim_status(mcp_client_started):
    out = call_tool(mcp_client_started, "check_claim_status", {"claim_id": "CLM-9003"})
    assert out["found"] is True
    assert out["status"] == "denied"
    assert "prior authorization" in out["denial_reason"].lower()


def test_mcp_estimate_patient_cost(mcp_client_started):
    out = call_tool(
        mcp_client_started,
        "estimate_patient_cost",
        {"member_id": "MBR-1002", "procedure_code": "70553", "billed_amount": 2400.0},
    )
    assert out["estimable"] is True
    assert out["patient_total"] == 1344.0


def test_mcp_submit_claim_is_idempotent(mcp_client_started):
    key = f"test-mcp-{uuid.uuid4().hex[:8]}"
    args = {
        "member_id": "MBR-1001",
        "procedure_code": "99213",
        "billed_amount": 250.0,
        "provider_npi": "NPI-1111111111",
        "idempotency_key": key,
        "confirmed": True,
    }
    first = call_tool(mcp_client_started, "submit_claim", args)
    assert first["submitted"] is True
    assert first["duplicate"] is False

    second = call_tool(mcp_client_started, "submit_claim", args)
    assert second["submitted"] is True
    assert second["duplicate"] is True
    assert second["claim_id"] == first["claim_id"]


def test_mcp_submit_claim_requires_confirmation(mcp_client_started):
    out = call_tool(
        mcp_client_started,
        "submit_claim",
        {
            "member_id": "MBR-1001",
            "procedure_code": "99213",
            "billed_amount": 250.0,
            "provider_npi": "NPI-1111111111",
            "idempotency_key": f"test-noconfirm-{uuid.uuid4().hex[:8]}",
            "confirmed": False,
        },
    )
    assert out["submitted"] is False
