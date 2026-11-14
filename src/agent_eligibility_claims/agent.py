"""The Strands agent, wired to the MCP server over stdio.

The agent itself holds no healthcare logic. It talks to the user, figures
out which tool to call, and explains what the tool returned. Every decision
— eligible or not, what a claim costs, whether a claim exists — is made by
the deterministic tools on the MCP server.

By default the agent runs against a real model (LiteLLM with
``openai/gpt-4o-mini``) and needs OPENAI_API_KEY — from the environment or
a ``.env`` file in the working directory. Pass ``provider="gemini"`` to
run it on the Gemini free tier instead (LiteLLM
``gemini/gemini-3.5-flash-lite``, needs GEMINI_API_KEY from Google AI
Studio — no credit card). The scripted mock model exists only for the
tests and the explicit offline demo (``--mock``); the real path never
silently falls back to it. If the key is missing, agent construction
fails with a clear message instead of pretending.

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
from pathlib import Path
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

# One entry per real-model provider. "model_id" is the LiteLLM model id;
# "key_env" is the environment variable the key comes from. Gemini uses
# gemini-3.5-flash-lite: it is on Google's free tier with the highest
# daily quota of the Flash line (about 500 requests/day reported in AI
# Studio, against about 20/day for plain Flash), and it is the model
# gemini-2.5-flash routes to after that ID stops serving new users on
# 2026-12-01. Model names rotate — check Google's pricing page before
# changing this.
PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "model_id": "openai/gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "key_hint": (
            "Get one at https://platform.openai.com/api-keys — the API is "
            "billed separately from a ChatGPT subscription."
        ),
    },
    "gemini": {
        "model_id": "gemini/gemini-3.5-flash-lite",
        "key_env": "GEMINI_API_KEY",
        "key_hint": (
            "Get a free key at Google AI Studio "
            "(https://aistudio.google.com) — no credit card needed."
        ),
    },
}


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


def load_dotenv(env_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment.

    Only sets keys that are not already set — the real environment always
    wins. Looks at ``.env`` in the working directory unless a path is
    given. No dependency, no surprises: lines starting with ``#`` and blank
    lines are skipped, and surrounding quotes are stripped.
    """
    path = env_path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def require_provider_key(provider: str = "openai") -> str:
    """Return the API key for a provider, or fail loudly if missing.

    The real agent path never falls back to the mock model — if there is
    no key, this raises with instructions instead of silently pretending
    to be a live agent.
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: "
            + ", ".join(sorted(PROVIDERS))
        )
    load_dotenv()
    key_env = PROVIDERS[provider]["key_env"]
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"{key_env} is not set. The agent runs against real "
            f"{provider} by default — set the key in your environment, or "
            f"copy .env.example to .env and put the key there (never commit "
            f".env). {PROVIDERS[provider]['key_hint']} For the offline demo "
            f"only, run with --mock."
        )
    return key


def require_openai_key() -> str:
    """Return OPENAI_API_KEY, or fail loudly if it is missing.

    Kept for backwards compatibility — new code should use
    :func:`require_provider_key`.
    """
    return require_provider_key("openai")


def require_gemini_key() -> str:
    """Return GEMINI_API_KEY, or fail loudly if it is missing."""
    return require_provider_key("gemini")


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
    model: Any | None = None,
    provider: str = "openai",
    callback_handler: Any | None = None,
) -> Iterator[Agent]:
    """Start the MCP server, load its tools, and yield a ready agent.

    With no ``model`` given, the agent runs against a real model through
    LiteLLM and requires that provider's key — see
    :func:`require_provider_key`. ``provider="openai"`` (the default)
    uses ``openai/gpt-4o-mini`` and needs OPENAI_API_KEY;
    ``provider="gemini"`` uses ``gemini/gemini-3.5-flash-lite`` on the
    free tier and needs GEMINI_API_KEY. Pass a model explicitly only for
    the scripted model in tests and the offline demo; the real path never
    silently falls back to mock.

    The MCP client is owned by the Strands Agent: constructing the agent
    starts the server subprocess (via the client's load_tools hook), so the
    tool list is printed after the agent is built and before any request.
    """
    logging.getLogger("strands").addFilter(_RedactPIIFilter())
    client = create_mcp_client()
    if model is None:
        require_provider_key(provider)
        agent_model: Any = LiteLLMModel(model_id=PROVIDERS[provider]["model_id"])
    else:
        agent_model = model
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
    """Convenience helper for tests and the offline demo: agent on the mock model."""
    return eligibility_agent(model=ScriptedModel())


__all__ = [
    "EXPECTED_TOOLS",
    "PROVIDERS",
    "SYSTEM_PROMPT",
    "ScriptedModel",
    "create_mcp_client",
    "eligibility_agent",
    "list_loaded_tools",
    "load_dotenv",
    "print_tool_list",
    "redact_pii",
    "require_gemini_key",
    "require_openai_key",
    "require_provider_key",
]
