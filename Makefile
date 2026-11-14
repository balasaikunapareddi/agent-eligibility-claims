.PHONY: install test demo demo-mock demo-live server

install:
	python3 -m venv .venv || true
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -v

# Real agent against OpenAI (LiteLLM, openai/gpt-4o-mini). Needs OPENAI_API_KEY.
demo:
	.venv/bin/python -m agent_eligibility_claims.demo

# Offline demo on the scripted model — no key, no network.
demo-mock:
	.venv/bin/python -m agent_eligibility_claims.demo --mock

# Kept for compatibility: live is now the default, so this is the same as demo.
demo-live:
	.venv/bin/python -m agent_eligibility_claims.demo

server:
	.venv/bin/python -m agent_eligibility_claims.mcp_server
