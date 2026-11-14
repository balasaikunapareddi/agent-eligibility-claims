"""Live smoke test — runs only when OPENAI_API_KEY is set.

This is the one test that spends real money (a few cents). Everything else
runs offline on the scripted model.
"""

import os

import pytest
from strands.models.litellm import LiteLLMModel

from agent_eligibility_claims.agent import eligibility_agent
from conftest import result_text

pytestmark = pytest.mark.skipif(
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
