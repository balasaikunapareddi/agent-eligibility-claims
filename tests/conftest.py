"""Shared fixtures: MCP server connections for the tests."""

import pytest

from agent_eligibility_claims.agent import create_mcp_client


@pytest.fixture(scope="session")
def mcp_client():
    """One MCP client for orchestration tests.

    Deliberately NOT started here: constructing a Strands Agent with this
    client starts the server itself (via the client's load_tools hook), and
    starting it twice raises. The agent fixture stops it after each test so
    the next test's agent starts a fresh server process.
    """
    yield create_mcp_client()


@pytest.fixture(scope="session")
def mcp_client_started():
    """A manually started MCP client, for tests that call tools directly.

    Separate from ``mcp_client``: direct calls need an already-running
    session, while the Agent-managed client must start lazily.
    """
    client = create_mcp_client()
    client.start()
    yield client
    client.stop(None, None, None)


def result_text(result) -> str:
    """Pull the final text out of an AgentResult."""
    message = getattr(result, "message", None) or {}
    content = message.get("content", []) if isinstance(message, dict) else []
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and "text" in block
    ]
    return "".join(parts).strip()
