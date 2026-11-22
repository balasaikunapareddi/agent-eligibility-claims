"""MCP server exposing the eligibility + claims tools.

Run it directly (stdio transport, for local dev and for the Strands agent)::

    python -m agent_eligibility_claims.mcp_server

Any MCP client can use the same server — Claude Code, the MCP Inspector,
or any other client that speaks stdio MCP. All the deciding happens in
``logic.py``; these tool functions are just the protocol-facing wrappers.

The server holds its claim ledger in memory, so every fresh start is a
clean slate seeded with the mock claims in ``data/claims.json``.
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from . import logic
from .prompts import MCP_SERVER_INSTRUCTIONS

logger = logging.getLogger(__name__)

server = MCPServer(
    name="eligibility-claims",
    description="Deterministic healthcare eligibility and claims tools (mock payer data).",
    instructions=MCP_SERVER_INSTRUCTIONS,
)


@server.tool()
def verify_eligibility(
    member_id: str, procedure_code: str, provider_npi: str = ""
) -> dict:
    """Check whether a member's coverage is active and covers a procedure.

    Returns a coverage verdict: active/terminated, in-network or not,
    deductible remaining, copay, coinsurance, and whether prior
    authorization is required. Never approves payment.
    """
    return logic.verify_eligibility(member_id, procedure_code, provider_npi)


@server.tool()
def check_claim_status(claim_id: str) -> dict:
    """Look up a claim by ID and return its status and adjudication detail."""
    return logic.check_claim_status(claim_id)


@server.tool()
def estimate_patient_cost(
    member_id: str, procedure_code: str, billed_amount: float
) -> dict:
    """Estimate what a member will owe for a procedure before it happens.

    Deterministic math: allowed amount is 80% of billed; the patient pays
    the remaining deductible first, then coinsurance on the rest, plus the
    office-visit copay for office visits.
    """
    return logic.estimate_patient_cost(member_id, procedure_code, billed_amount)


@server.tool()
def submit_claim(
    member_id: str,
    procedure_code: str,
    billed_amount: float,
    provider_npi: str,
    idempotency_key: str,
    confirmed: bool = False,
) -> dict:
    """Submit a claim to the payer. Idempotent and requires confirmation.

    The same idempotency key twice returns the original claim instead of
    creating a duplicate. Set confirmed=true only after the user has
    explicitly confirmed the member, procedure, and billed amount.
    """
    return logic.submit_claim(
        member_id, procedure_code, billed_amount, provider_npi, idempotency_key, confirmed
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
