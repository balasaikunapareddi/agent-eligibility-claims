"""Live smoke tests — each runs only when its provider key is set.

These are the only tests that call a real model. The OpenAI one spends
a few cents; the Gemini one runs on the free tier.
"""

import os

import pytest
from strands.models.litellm import LiteLLMModel

from agent_eligibility_claims.agent import eligibility_agent
from conftest import result_text


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; live test skipped",
)
def test_live_openai_smoke():
    with eligibility_agent(model=LiteLLMModel(model_id="openai/gpt-4o-mini")) as agent:
        result = agent(
            "Check eligibility for member MBR-1001 for procedure 99213."
        )
    text = result_text(result)
    assert text, "live agent returned no text"
    assert "eligib" in text.lower()


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; live Gemini test skipped",
)
def test_live_gemini_smoke():
    """Same smoke test on the Gemini free tier — free, no card needed."""
    with eligibility_agent(
        model=LiteLLMModel(model_id="gemini/gemini-3.5-flash-lite")
    ) as agent:
        result = agent(
            "Check eligibility for member MBR-1001 for procedure 99213."
        )
    text = result_text(result)
    assert text, "live agent returned no text"
    assert "eligib" in text.lower()
