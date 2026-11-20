.PHONY: install test demo demo-gemini demo-mock demo-live server web

install:
	python3 -m venv .venv || true
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -v

# Real agent against OpenAI (LiteLLM, openai/gpt-4o-mini). Needs OPENAI_API_KEY.
demo:
	.venv/bin/python -m agent_eligibility_claims.demo

# Real agent against Gemini free tier (LiteLLM, gemini/gemini-3.5-flash-lite).
# Needs GEMINI_API_KEY — free key from Google AI Studio, no card needed.
demo-gemini:
	.venv/bin/python -m agent_eligibility_claims.demo --provider gemini

# Offline demo on the scripted model — no key, no network.
demo-mock:
	.venv/bin/python -m agent_eligibility_claims.demo --mock

# Kept for compatibility: live is now the default, so this is the same as demo.
demo-live:
	.venv/bin/python -m agent_eligibility_claims.demo

server:
	.venv/bin/python -m agent_eligibility_claims.mcp_server

# Chat UI in the browser (scripted mock by default; set GEMINI_API_KEY for live).
web:
	.venv/bin/uvicorn web.app:app --reload
