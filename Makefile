.PHONY: install test demo demo-live server

install:
	python3 -m venv .venv || true
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -v

demo:
	.venv/bin/python -m agent_eligibility_claims.demo

demo-live:
	.venv/bin/python -m agent_eligibility_claims.demo --live

server:
	.venv/bin/python -m agent_eligibility_claims.mcp_server
