"""Deterministic business logic for the eligibility + claims agent.

This module is deliberately boring: no LLM, no network, no randomness.
Every decision the agent makes about coverage, cost, or claim state comes
from here, so the same inputs always give the same outputs. The MCP server
in ``mcp_server.py`` is a thin wrapper around these functions, and the tests
exercise them both directly and through the MCP protocol.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

ALLOWED_AMOUNT_RATIO = 0.80  # in-network allowed amount is 80% of billed
FIRST_CLAIM_SEQ = 9100


def _load_json(name: str) -> Any:
    with (resources.files("agent_eligibility_claims") / "data" / name).open() as f:
        return json.load(f)


def load_members() -> dict[str, dict[str, Any]]:
    data = _load_json("members.json")
    return {m["member_id"]: m for m in data["members"]}


def load_procedures() -> dict[str, dict[str, Any]]:
    return _load_json("members.json")["procedures"]


def load_seed_claims() -> list[dict[str, Any]]:
    return _load_json("claims.json")["claims"]


MEMBERS = load_members()
PROCEDURES = load_procedures()


class ClaimStore:
    """In-memory claim ledger with idempotency keys.

    Lives in the MCP server process, so each server start is a fresh ledger
    seeded with the mock claims. Submitting the same idempotency key twice
    returns the original claim instead of creating a second one.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.claims: dict[str, dict[str, Any]] = {
            c["claim_id"]: dict(c) for c in load_seed_claims()
        }
        self.idempotency_keys: dict[str, str] = {}
        self._next_seq = FIRST_CLAIM_SEQ

    def next_claim_id(self) -> str:
        self._next_seq += 1
        return f"CLM-{self._next_seq}"


STORE = ClaimStore()


def verify_eligibility(
    member_id: str, procedure_code: str, provider_npi: str = ""
) -> dict[str, Any]:
    """Decide coverage for one member + procedure. Pure function of the data."""
    member = MEMBERS.get(member_id.strip().upper())
    if member is None:
        return {
            "eligible": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "reason": f"Member {member_id} not found in the payer system.",
        }
    procedure = PROCEDURES.get(procedure_code.strip())
    if procedure is None:
        return {
            "eligible": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "reason": f"Unknown procedure code {procedure_code}.",
        }
    if member["coverage_status"] != "active":
        return {
            "eligible": False,
            "member_id": member_id,
            "member_name": member["name"],
            "plan": member["plan"],
            "procedure_code": procedure_code,
            "procedure_description": procedure["description"],
            "coverage_status": member["coverage_status"],
            "coverage_end": member["coverage_end"],
            "reason": (
                f"Coverage is {member['coverage_status']} as of "
                f"{member['coverage_end']}. "
                f"{member.get('termination_reason', '')}".strip()
            ),
        }

    in_network = not provider_npi or provider_npi in member["in_network_providers"]
    prior_auth = (
        procedure_code in member["prior_auth_required_for"]
        or procedure["requires_prior_auth"]
    )
    deductible_remaining = round(
        member["deductible_total"] - member["deductible_met"], 2
    )
    return {
        "eligible": bool(in_network),
        "member_id": member_id,
        "member_name": member["name"],
        "plan": member["plan"],
        "procedure_code": procedure_code,
        "procedure_description": procedure["description"],
        "coverage_status": "active",
        "coverage_end": member["coverage_end"],
        "in_network": in_network,
        "deductible_remaining": deductible_remaining,
        "copay_office_visit": member["copay_office_visit"],
        "coinsurance_rate": member["coinsurance_rate"],
        "prior_auth_required": prior_auth,
        "reason": (
            "Eligible: active coverage, provider is in-network."
            if in_network
            else "Not eligible: provider is out-of-network for this plan."
        ),
    }


def check_claim_status(claim_id: str) -> dict[str, Any]:
    """Look up one claim. Returns the adjudication detail, or a not-found error."""
    claim = STORE.claims.get(claim_id.strip().upper())
    if claim is None:
        return {
            "found": False,
            "claim_id": claim_id,
            "reason": f"Claim {claim_id} not found.",
        }
    return {"found": True, **dict(claim)}


def estimate_patient_cost(
    member_id: str, procedure_code: str, billed_amount: float
) -> dict[str, Any]:
    """Deterministic patient-cost math.

    allowed = 80% of billed. The patient first pays the remaining deductible
    (up to the allowed amount), then coinsurance on what is left of the
    allowed amount, plus the office-visit copay for office visits. Anything
    billed above the allowed amount is a provider write-off in-network and is
    never the patient's problem.
    """
    eligibility = verify_eligibility(member_id, procedure_code)
    if not eligibility.get("eligible"):
        return {
            "estimable": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "billed_amount": billed_amount,
            "reason": eligibility.get("reason", "Not eligible."),
        }
    if billed_amount <= 0:
        return {
            "estimable": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "billed_amount": billed_amount,
            "reason": "Billed amount must be greater than zero.",
        }

    allowed = round(billed_amount * ALLOWED_AMOUNT_RATIO, 2)
    deductible_applied = min(eligibility["deductible_remaining"], allowed)
    coinsurance_base = round(allowed - deductible_applied, 2)
    coinsurance_owed = round(
        coinsurance_base * eligibility["coinsurance_rate"], 2
    )
    copay = (
        eligibility["copay_office_visit"] if procedure_code == "99213" else 0.0
    )
    patient_total = round(deductible_applied + coinsurance_owed + copay, 2)
    return {
        "estimable": True,
        "member_id": member_id,
        "member_name": eligibility["member_name"],
        "plan": eligibility["plan"],
        "procedure_code": procedure_code,
        "procedure_description": eligibility["procedure_description"],
        "billed_amount": round(billed_amount, 2),
        "allowed_amount": allowed,
        "deductible_remaining_before": eligibility["deductible_remaining"],
        "deductible_applied": deductible_applied,
        "coinsurance_rate": eligibility["coinsurance_rate"],
        "coinsurance_owed": coinsurance_owed,
        "copay": copay,
        "patient_total": patient_total,
        "plan_pays": round(allowed - deductible_applied - coinsurance_owed, 2),
        "provider_write_off": round(billed_amount - allowed, 2),
    }


def submit_claim(
    member_id: str,
    procedure_code: str,
    billed_amount: float,
    provider_npi: str,
    idempotency_key: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """File a claim. Idempotent on ``idempotency_key`` and refused without
    explicit confirmation — the agent must ask the user to confirm first."""
    if not confirmed:
        return {
            "submitted": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "reason": (
                "Confirmation required: claim submission is a commitment. "
                "Ask the user to confirm the member, procedure, and billed "
                "amount, then call again with confirmed=true."
            ),
        }
    key = idempotency_key.strip()
    if not key:
        return {
            "submitted": False,
            "reason": "An idempotency key is required to submit a claim.",
        }
    if key in STORE.idempotency_keys:
        original = STORE.idempotency_keys[key]
        return {
            "submitted": True,
            "duplicate": True,
            "claim_id": original,
            "idempotency_key": key,
            "status": STORE.claims[original]["status"],
            "reason": (
                "This idempotency key was already used. No new claim was "
                f"created; returning the original claim {original}."
            ),
        }

    eligibility = verify_eligibility(member_id, procedure_code, provider_npi)
    if not eligibility.get("eligible"):
        return {
            "submitted": False,
            "member_id": member_id,
            "procedure_code": procedure_code,
            "reason": eligibility.get("reason", "Not eligible."),
        }
    if billed_amount <= 0:
        return {
            "submitted": False,
            "reason": "Billed amount must be greater than zero.",
        }

    claim_id = STORE.next_claim_id()
    claim = {
        "claim_id": claim_id,
        "member_id": member_id.strip().upper(),
        "procedure_code": procedure_code,
        "provider_npi": provider_npi,
        "billed_amount": round(billed_amount, 2),
        "status": "submitted",
        "submitted_date": "2026-11-13",
        "adjudicated_date": None,
        "allowed_amount": None,
        "paid_amount": None,
        "patient_responsibility": None,
        "denial_reason": None,
        "notes": "Submitted to payer; awaiting adjudication.",
    }
    STORE.claims[claim_id] = claim
    STORE.idempotency_keys[key] = claim_id
    return {
        "submitted": True,
        "duplicate": False,
        "claim_id": claim_id,
        "idempotency_key": key,
        "status": "submitted",
        "member_id": claim["member_id"],
        "procedure_code": procedure_code,
        "billed_amount": claim["billed_amount"],
        "reason": "Claim submitted successfully.",
    }
