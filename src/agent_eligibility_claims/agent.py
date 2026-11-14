"""The Strands agent, wired to the MCP server over stdio.

The agent itself holds no healthcare logic. It talks to the user, figures
out which tool to call, and explains what the tool returned. Every decision
— eligible or not, what a claim costs, whether a claim exists — is made by
the deterministic tools on the MCP server.

Startup always prints the tools the agent loaded from the MCP server, so
you can see exactly what it is allowed to call before it takes any request.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands.tools.mcp import MCPClient

from .scripted_model import ScriptedModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a healthcare eligibility and claims assistant. You help
front-desk staff and patients understand coverage, claim status, and costs.

Rules you must follow:
- Use your tools to look things up. Never guess at coverage, a claim
  status, or a cost — call the tool and report what it returns.
- You explain results. You never approve payment, never decide a claim
  outcome, and never change coverage. Those decisions belong to the payer
  and the tools.
- Before calling submit_claim, ask the user to confirm the member,
  procedure, and billed amount, and only call it with confirmed=true after
  they say yes. If they have not confirmed, do not submit.
- Member IDs and dates of birth are sensitive. Do not paste them into logs
  or repeat them unless the user asked for them.
- All data is synthetic mock data for demonstration. Say so if asked.
"""

EXPECTED_TOOLS = [
    "verify_eligibility",
    "check_claim_status",
    "estimate_patient_cost",
    "submit_claim",
]


def redact_pii(text: str) -> str:
    """Mask member IDs and the mock members' names in log output."""
    text = re.sub(r"MBR-\d+", "MBR-****", text)
    for name in ("Asha Sharma", "Jay Patel", "Ravi Iyer", "Sita Rao"):
        text = text.replace(name, "[member]")
    return text


class _RedactPIIFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)
        if record.args:
            try:
                record.args = tuple(
                    redact_pii(a) if isinstance(a, str) else a for a in record.args
                )
            except TypeError:
                pass
        return True


def create_mcp_client() -> MCPClient:
    """Build the MCP client that spawns the eligibility-claims server."""

    def _transport():
        return stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-m", "agent_eligibility_claims.mcp_server"],
                env={**os.environ},
            )
        )

    return MCPClient(_transport)


def list_loaded_tools(client: MCPClient) -> list[tuple[str, str]]:
    """Return (name, one-line description) for every tool on the server."""
    tools = client.list_tools_sync()
    out = []
    for tool in tools:
        description = (tool.tool_spec.get("description") or "").strip()
        one_line = description.splitlines()[0] if description else "(no description)"
        out.append((tool.tool_name, one_line))
    return out


def print_tool_list(tools: list[tuple[str, str]]) -> None:
    print(f"Loaded {len(tools)} tools from the MCP server:")
    for name, description in tools:
        print(f"  - {name}: {description}")


@contextmanager
def eligibility_agent(
    model: Any | None = None, callback_handler: Any | None = None
) -> Iterator[Agent]:
    """Start the MCP server, load its tools, and yield a ready agent.

    The MCP client is owned by the Strands Agent: constructing the agent
    starts the server subprocess (via the client's load_tools hook), so the
    tool list is printed after the agent is built and before any request.
    """
    logging.getLogger("strands").addFilter(_RedactPIIFilter())
    client = create_mcp_client()
    agent_model = model or LiteLLMModel(model_id="openai/gpt-4o-mini")
    agent = Agent(
        model=agent_model,
        tools=[client],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=callback_handler,
    )
    try:
        print_tool_list(list_loaded_tools(client))
        yield agent
    finally:
        client.stop(None, None, None)


def build_scripted_agent() -> Any:
    """Convenience helper for scripts: an agent on the scripted model."""
    return eligibility_agent(model=ScriptedModel())


__all__ = [
    "EXPECTED_TOOLS",
    "SYSTEM_PROMPT",
    "ScriptedModel",
    "create_mcp_client",
    "eligibility_agent",
    "list_loaded_tools",
    "print_tool_list",
    "redact_pii",
]
