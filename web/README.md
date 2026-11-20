# Eligibility-claims web UI

The chat front door: ask the agent a coverage, claim-status, or
cost question in the browser and watch it call the four
deterministic MCP tools — every tool call is shown with its inputs
and full JSON result. No install beyond pip, no API key.

## Run it locally

```bash
pip install -r web/requirements.txt
uvicorn web.app:app --reload        # http://127.0.0.1:8000
```

(from the repo root — `web` is imported as a package; the app puts
`src/` on the path itself. `make web` does the same thing.)

## Test it

```bash
pip install httpx pytest
python3 -m pytest web -q
```

## How it works

- `app.py` — FastAPI app: home page (chat panel + sample
  questions), `POST /chat`, `POST /confirm`, `POST /reset`,
  `/health`. Server-rendered HTML with inline CSS (same look as
  the cookbook / clinical-inference / scheduler / booking UIs);
  no build step, no frontend framework.
- One long-lived agent per process, built lazily on the first
  question via the repo's own `eligibility_agent(...)` — the same
  Strands agent + MCP-server-over-stdio wiring as the CLI demo.
  Turns are serialized with a lock. Because the MCP server (and
  its in-memory claim ledger) stays up, submitted claims and
  their idempotency keys persist across turns; **Reset** starts a
  fresh agent with a fresh ledger.
- Tool calls are read back out of the agent's message history
  after each turn and rendered with a one-line digest plus the
  full JSON result — the tools are the point of this repo, so
  they are never hidden.
- `submit_claim` keeps its confirmation behavior: if the tool
  refuses for lack of confirmation, the page shows a confirm
  panel with the parsed member / procedure / amount / provider /
  idempotency key. Only the explicit **Yes — submit this claim**
  button sends the confirmed follow-up. Submitting again with
  the same key returns the original claim (idempotent).
- Mock mode by default: the scripted model, deterministic,
  offline, no key. If `GEMINI_API_KEY` is set in the environment
  (or a local `.env`), the agent runs live on the Gemini free
  tier instead; if a live call fails, the app rebuilds on the
  scripted model and says so on the page. The current mode is
  shown as a badge above the chat box.
- The agent's system prompt is displayed verbatim in a
  collapsible section, per the standing rule that UIs show
  their prompts.
- Failures are shown, not hidden: agent errors render as a
  friendly banner (never a 500), and the conversation so far
  stays on the page.

## Deploy

`render.yaml` at the repo root is a Render Blueprint (free tier,
`plan: free`): build installs `web/requirements.txt`, start runs
uvicorn on `$PORT`, health check is `/health`. Nothing to
configure in the dashboard — no keys, no database. Optionally
set `GEMINI_API_KEY` there to enable live mode.

Demo on synthetic mock data only — no real patient information.
