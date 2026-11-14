"""Demo: the eligibility + claims agent, end to end, in one command.

By default it runs against a real model — LiteLLM with
``openai/gpt-4o-mini`` — and needs OPENAI_API_KEY (environment or a
``.env`` file). Pass ``--provider gemini`` (or ``--gemini``) to run the
same demo free on the Gemini free tier — LiteLLM with
``gemini/gemini-3.5-flash-lite`` — which needs GEMINI_API_KEY from
Google AI Studio, no credit card. Pass ``--mock`` for the offline demo
on the scripted model, which needs no key and no network. There is no
silent fallback: without the right key the real path fails with a clear
message telling you to set one.

    python -m agent_eligibility_claims.demo                    # real OpenAI
    python -m agent_eligibility_claims.demo --provider gemini  # real Gemini, free
    python -m agent_eligibility_claims.demo --mock             # offline, scripted
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import PROVIDERS, eligibility_agent, require_provider_key
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
        "--mock",
        action="store_true",
        help="Run the offline demo on the scripted model (no key, no network).",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="openai",
        help="Which real model to run on: openai (default) or gemini (free tier).",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Shorthand for --provider gemini (free tier, no card needed).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Accepted for compatibility — live OpenAI is now the default.",
    )
    args = parser.parse_args(argv)
    provider = "gemini" if args.gemini else args.provider

    if args.mock:
        model = ScriptedModel()
        print("Running on the scripted model (offline, deterministic).")
        print("Mock is for tests and offline use only — the real agent runs on OpenAI or Gemini.")
    else:
        # Fails with a clear message when the provider's key is missing;
        # never falls back to the mock model or the other provider.
        require_provider_key(provider)
        model = None
        if provider == "gemini":
            print("Running LIVE with LiteLLM (gemini/gemini-3.5-flash-lite) — free tier.")
        else:
            print("Running LIVE with LiteLLM (openai/gpt-4o-mini).")

    print("All data is synthetic mock data — no real patient information.\n")

    with eligibility_agent(
        model=model, provider=provider, callback_handler=_tool_call_callback
    ) as agent:
        for title, prompt in SCENARIOS:
            print(f"\n### {title}")
            print(f"You: {prompt}")
            result = agent(prompt)
            print(f"Agent: {result_text(result)}")
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
