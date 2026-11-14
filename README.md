# agent-eligibility-claims

A healthcare agent that answers the questions every front desk and billing
team asks all day: *is this patient covered for this procedure, what's the
claim doing, and what will they owe?*

The boring part is the point. Eligibility rules, deductibles, coinsurance —
that math should not depend on how a language model happens to feel today.
So this agent is built the other way round: the model talks, and
deterministic code decides. The model figures out what the person is asking
for, the tools look it up in the mock payer data and do the arithmetic, and
the model explains the answer back in plain language. Nothing about coverage
or money is ever left to the model's imagination.

## What it does

Four tools, all of them deterministic, all of them served over MCP:

- `verify_eligibility` — member ID + procedure → covered or not, in-network
  or not, deductible remaining, copay, whether prior auth is required.
- `check_claim_status` — claim ID → paid / pending / denied / in review,
  with the adjudication detail (including the denial reason when there is one).
- `estimate_patient_cost` — member + procedure + billed amount → allowed
  amount, deductible applied, coinsurance, copay, what the patient owes, what
  the plan pays. Straight arithmetic: allowed is 80% of billed, deductible
  comes first, then coinsurance on what's left, plus the office-visit copay
  when one applies.
- `submit_claim` — submits a claim, but only with an explicit confirmation
  from the user, and only once per idempotency key. Submit the same claim
  twice with the same key and the second call is a no-op that returns the
  original claim ID.

Guardrails, baked into the system prompt: the agent looks things up and
explains them, it never approves a payment, and claim submission always
needs a human saying yes first. Member IDs are redacted from the logs.

## How it's put together

```mermaid
flowchart LR
    U[You] --> A[Strands agent\nLiteLLM / scripted model]
    A -->|stdio| M[MCP server\n4 deterministic tools]
    M --> L[logic.py\ncoverage + cost math]
    M --> D[(mock payer data\nmembers.json, claims.json)]
    A -->|explains| U
```

The agent is a single Strands agent. The tools live in a separate MCP
server process (`agent_eligibility_claims.mcp_server`), which the agent
connects to over stdio when it starts up. The model never touches the
payer data directly — it can only call the tools, and the tools do the
deciding.

When the agent starts, it prints the tools it loaded, so you can see
exactly what it is allowed to call before it takes any request:

```
Loaded 4 tools from the MCP server:
  - verify_eligibility: Check whether a member's coverage is active and covers a procedure.
  - check_claim_status: Look up a claim by ID and return its status and adjudication detail.
  - estimate_patient_cost: Estimate what a member will owe for a procedure before it happens.
  - submit_claim: Submit a claim to the payer. Idempotent and requires confirmation.
```

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make demo        # or: .venv/bin/python -m agent_eligibility_claims.demo
```

That's it — the demo runs on a scripted model, so it needs no API key and
no network. It walks through six scenarios against the mock data:

```
### 1. Eligibility check that passes
You: Check eligibility for member MBR-1001 for procedure 99213.
Agent: Member MBR-1001 is eligible for 99213 (Office visit, established
patient, 15 min): active Acme Health PPO Gold, in-network, $0.0 deductible
remaining, $25.0 copay. Prior authorization required: no.

### 2. Eligibility check that fails (terminated coverage)
You: Check eligibility for member MBR-1003 for procedure 70553.
Agent: Member MBR-1003 is NOT eligible for 70553: Coverage is terminated
as of 2026-09-30. Employment ended; COBRA not elected

### 3. Claim status lookup (denied claim)
You: What is the status of claim CLM-9003?
Agent: Claim CLM-9003 is denied. Billed $1800.0. Denial reason: Prior
authorization required but not obtained (CARC 197).

### 4. Patient cost estimate
You: Estimate the patient cost for member MBR-1002, procedure 70553,
billed $2400.00.
Agent: Estimated patient cost for 70553: $1344.0 (allowed $1920.0 on
$2400.0 billed; deductible $1200.0, coinsurance $144.0, copay $0.0).
The plan pays $576.0.

### 5a. Submit a claim (confirmed)
Agent: Claim submitted: CLM-9101, status submitted.

### 5b. Submit the same claim again (same idempotency key)
Agent: Already submitted — this idempotency key was used before, so no
new claim was created. Original claim: CLM-9101 (status submitted).
```

## The mock data

Everything is synthetic — no real patient information anywhere in this
repo. Four members, each in a different coverage state:

| Member | Plan | State |
| --- | --- | --- |
| MBR-1001 — Asha Sharma | Acme Health PPO Gold | Active, deductible met |
| MBR-1002 — Jay Patel | Acme Health PPO Silver | Active, $1,200 deductible left |
| MBR-1003 — Ravi Iyer | Acme Health PPO Gold | Terminated 2026-09-30 |
| MBR-1004 — Sita Rao | Acme Health HMO | Active, in-network only |

And four claims in four states: CLM-9001 paid, CLM-9002 pending,
CLM-9003 denied (prior auth, CARC 197), CLM-9004 in review. The data
lives in `src/agent_eligibility_claims/data/members.json` and
`claims.json` — edit those files to try different scenarios.

## Using the MCP server from any client

The tools aren't locked to this agent. The server is a plain MCP server
(MCP Python SDK), so anything that speaks MCP — Claude Code, other agent
frameworks, an MCP inspector — can use the same four tools:

```bash
# Start it standalone (stdio transport):
.venv/bin/python -m agent_eligibility_claims.mcp_server
```

Point your MCP client at that command with stdio transport. The server
exposes exactly four tools — `verify_eligibility`, `check_claim_status`,
`estimate_patient_cost`, `submit_claim` — with JSON schemas generated
from their function signatures. The agent in this repo connects the same
way: it spawns the server as a subprocess and talks to it through
Strands' `MCPClient` (`strands.tools.mcp.MCPClient`).

One thing to know: the server keeps its claim ledger in memory, so
submitted claims (and their idempotency keys) only live as long as the
server process does. Restart it and the ledger resets to the four seed
claims.

## Running it live

The demo and the tests run on a scripted model by default — deterministic,
free, offline. To run against a real model instead:

```bash
cp .env.example .env   # then put your key in .env
export OPENAI_API_KEY=sk-...
make demo-live         # or: python -m agent_eligibility_claims.demo --live
```

That swaps in LiteLLM with `openai/gpt-4o-mini`. The tools, the guardrails,
and the scenarios are identical — only the model changes. Never commit a
real key: `.env` is gitignored, and `.env.example` is the only key-shaped
file in the repo.

## Running the tests

```bash
make test   # or: .venv/bin/python -m pytest -v
```

25 tests, no key needed:

- `tests/test_logic.py` — the deterministic logic directly: eligibility
  verdicts, claim statuses, the cost math, idempotency, the confirmation
  gate.
- `tests/test_mcp_tools.py` — the tools called over the real MCP protocol
  through a client, including a test that the agent loads exactly the four
  expected tools.
- `tests/test_orchestration.py` — the full agent with the scripted model:
  the five demo scenarios end to end, including submitting the same claim
  twice.
- `tests/test_live.py` — one smoke test against OpenAI. It only runs when
  `OPENAI_API_KEY` is set and skips otherwise.

## Layout

```
src/agent_eligibility_claims/
  agent.py            Strands agent, system prompt, MCP client, tool listing
  mcp_server.py       the MCP server (the four tools)
  logic.py            the deterministic rules — no model, no network
  scripted_model.py   rule-based stand-in model for tests and the offline demo
  demo.py             the six-scenario demo CLI
  data/               mock members + claims (synthetic)
tests/
  test_logic.py  test_mcp_tools.py  test_orchestration.py  test_live.py
```

## Why this exists

Healthcare eligibility and claims are a good test of whether an agent
architecture is actually trustworthy: the stakes are real money and real
people's coverage, and a confident wrong answer is worse than no answer.
The pattern here — model talks, code decides, every money-moving action
needs a human yes and an idempotency key — is the one I'd want in front
of a real payer system. This repo is the mock-data version of that
pattern, runnable in one command.
