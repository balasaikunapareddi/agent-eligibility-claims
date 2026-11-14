"""Demo: the eligibility + claims agent, end to end, in one command.

By default it runs on the scripted model — no API key, no network, same
output every time. Pass ``--live`` to run the same five scenarios against
a real model through LiteLLM (needs OPENAI_API_KEY)::

    python -m agent_eligibility_claims.demo
    python -m agent_eligibility_claims.demo --live
"""

from __future__ import annotations

import argparse
import json
import sys

from strands.models.litellm import LiteLLMModel

from .agent import eligibility_agent
from .scripted_model import ScriptedModel

SCENARIOS: list[tuple[str, str]] = [
    (
        "1. Eligibility check that passes",
        "Check eligibility for member MBR-1001 for procedure 99213.",
    ),
    (
        "2. Eligibility check that fails (terminated coverage)",
        "Check eligibility for member MBR-1003 for procedure 70553.",
    ),
    (
        "3. Claim status lookup (denied claim)",
        "What is the status of claim CLM-9003?",
    ),
    (
        "4. Patient cost estimate",
        "Estimate the patient cost for member MBR-1002, procedure 70553, billed $2400.00.",
    ),
    (
        "5a. Submit a claim (confirmed)",
        "Submit a claim for member MBR-1001, procedure 99213, billed $250.00, "
        "provider NPI-1111111111, idempotency key demo-2026-001. I confirm this submission.",
    ),
    (
        "5b. Submit the same claim again (same idempotency key)",
        "Submit a claim for member MBR-1001, procedure 99213, billed $250.00, "
        "provider NPI-1111111111, idempotency key demo-2026-001. I confirm this submission.",
    ),
]


def _tool_call_callback(**kwargs: object) -> None:
    """Print tool calls as they happen; the final text is printed by main."""
    use = kwargs.get("current_tool_use")
    if isinstance(use, dict):
        raw_input = use.get("input", {})
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except json.JSONDecodeError:
                pass
        print(f"  → tool: {use.get('name')}({json.dumps(raw_input)})")


def result_text(result: object) -> str:
    """Pull the final text out of an AgentResult."""
    message = getattr(result, "message", None) or {}
    content = message.get("content", []) if isinstance(message, dict) else []
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and "text" in block
    ]
    text = "".join(parts).strip()
    return text or str(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eligibility + claims agent demo")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use LiteLLM/OpenAI (gpt-4o-mini) instead of the scripted model. Needs OPENAI_API_KEY.",
    )
    args = parser.parse_args(argv)

    if args.live:
        model = LiteLLMModel(model_id="openai/gpt-4o-mini")
        print("Running LIVE with LiteLLM (openai/gpt-4o-mini).")
    else:
        model = ScriptedModel()
        print("Running on the scripted model (offline, deterministic).")

    print("All data is synthetic mock data — no real patient information.\n")

    with eligibility_agent(model=model, callback_handler=_tool_call_callback) as agent:
        for title, prompt in SCENARIOS:
            print(f"\n### {title}")
            print(f"You: {prompt}")
            result = agent(prompt)
            print(f"Agent: {result_text(result)}")
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
