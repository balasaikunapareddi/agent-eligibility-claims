"""agent-eligibility-claims — chat with the agent in your browser.

A thin FastAPI front door over the same Strands agent the CLI demo
runs: type a question (or click a sample), and the agent answers by
calling the four deterministic MCP tools — verify_eligibility,
check_claim_status, estimate_patient_cost, submit_claim. Every tool
call is shown with its inputs and result, because the tools are the
point: the model talks, deterministic code decides.

    pip install -r web/requirements.txt
    uvicorn web.app:app --reload        # http://127.0.0.1:8000

Mock mode by default: the scripted model, deterministic, offline,
no API key. If GEMINI_API_KEY is set in the environment, the agent
runs live on the Gemini free tier instead (LiteLLM
``gemini/gemini-3.5-flash-lite``), with an automatic fall back to the
scripted model if a live call fails. One agent (and its MCP server
subprocess) is kept for the life of the process, so submitted claims
and their idempotency keys persist across turns — "Reset" starts a
fresh agent with a fresh ledger.

Demo on synthetic mock data only — no real patient information.
"""

from __future__ import annotations

import html
import json
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The agent package lives in src/ (see pyproject.toml); put it on the
# path so `uvicorn web.app:app` works from the repo root with no
# install step, same trick as the cookbook / clinical-inference UIs.
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi import FastAPI, Form  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from agent_eligibility_claims import logic  # noqa: E402
from agent_eligibility_claims.agent import (  # noqa: E402
    eligibility_agent,
    load_dotenv,
)
from agent_eligibility_claims.prompts import SYSTEM_PROMPT  # noqa: E402
from agent_eligibility_claims.demo import result_text  # noqa: E402
from agent_eligibility_claims.scripted_model import ScriptedModel  # noqa: E402

GITHUB_URL = "https://github.com/balasaikunapareddi/agent-eligibility-claims"
MAX_TURNS = 20  # keep the rendered history bounded

SAMPLE_QUESTIONS = [
    (
        "Eligibility — covered",
        "Check eligibility for member MBR-1001 for procedure 99213.",
    ),
    (
        "Eligibility — terminated",
        "Check eligibility for member MBR-1003 for procedure 70553.",
    ),
    ("Claim status — denied", "What is the status of claim CLM-9003?"),
    (
        "Cost estimate",
        "Estimate the patient cost for member MBR-1002, procedure 70553, "
        "billed $2400.00.",
    ),
    (
        "Submit a claim (asks first)",
        "Submit a claim for member MBR-1001, procedure 99213, billed "
        "$250.00, provider NPI-1111111111, idempotency key web-demo-001.",
    ),
]

app = FastAPI(title="agent-eligibility-claims")

STYLE = """
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; color: #1a1a1a; background: #f6f6f4; }
  header { background: #173a2f; color: #fff; padding: 14px 20px; }
  header h1 { margin: 0; font-size: 18px; }
  header h1 a { color: #fff; text-decoration: none; }
  header p { margin: 4px 0 0; font-size: 13px; opacity: .8; }
  main { max-width: 960px; margin: 0 auto; padding: 20px 16px 40px; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 14px; }
  .panel h2 { margin: 0 0 10px; font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: #555; }
  .run-panel { border: 2px solid #173a2f; background: #fbfdfc; }
  .run-panel h2 { color: #173a2f; font-size: 15px; }
  .warn { background: #fdf6e3; border: 1px solid #e8d9a0; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 13px; line-height: 1.6; }
  .error { background: #fdecea; border: 1px solid #e5b4ae; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 13px; line-height: 1.6; }
  pre.output { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  textarea { width: 100%; min-height: 64px; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; font-family: inherit; background: #fff; resize: vertical; }
  button[type=submit] { padding: 9px 16px; background: #173a2f; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 8px; }
  button.run-big { display: block; width: 100%; padding: 14px 16px; font-size: 17px; font-weight: 700; border-radius: 8px; margin-top: 10px; }
  button.run-big:hover { background: #1f4d3d; }
  button.altbtn { background: #fff; color: #173a2f; border: 2px solid #173a2f; }
  .samples { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
  .samples form { margin: 0; }
  .samples button { margin-top: 0; padding: 6px 10px; background: #fafafa; color: #173a2f; border: 1px solid #cfded6; font-size: 12px; }
  .hint { font-size: 12px; color: #888; margin-top: 8px; line-height: 1.5; }
  .meta { font-size: 13px; color: #555; line-height: 1.6; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f0f0ec; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
  details summary { cursor: pointer; font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: #555; }
  details pre.output { margin-top: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
  th { color: #555; font-weight: 600; }
  .badge { display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #173a2f; background: #eef4f1; border: 1px solid #cfded6; border-radius: 10px; padding: 2px 8px; }
  .turn { margin-bottom: 14px; }
  .turn .who { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #888; margin-bottom: 3px; }
  .turn .user { background: #eef4f1; border: 1px solid #cfded6; border-radius: 8px; padding: 10px 12px; font-size: 14px; line-height: 1.5; }
  .turn .agent { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; font-size: 14px; line-height: 1.6; margin-top: 6px; }
  .toolcall { background: #fbfdfc; border: 1px solid #cfded6; border-radius: 8px; padding: 10px 12px; margin-top: 6px; }
  .toolcall .call { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; word-break: break-word; }
  .toolcall .digest { font-size: 13px; color: #333; margin-top: 4px; line-height: 1.5; }
  .toolcall details { margin-top: 6px; }
  .toolcall details summary { font-size: 11px; text-transform: none; letter-spacing: 0; color: #888; }
  .confirm-panel { border: 2px solid #b58900; background: #fdf6e3; }
  .confirm-panel h2 { color: #7a5c00; }
  .btnrow { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
  .btnrow form { margin: 0; }
  .btnrow button { width: 100%; margin-top: 0; }
  .resetrow { text-align: right; }
  .resetrow button { background: #fff; color: #173a2f; border: 1px solid #cfded6; font-size: 12px; padding: 6px 10px; }
"""

DISCLAIMER = (
    '<div class="warn"><strong>Demo on synthetic mock data.</strong> '
    "No real patient information, no real payer — four fictional "
    "members, five procedure codes, four seed claims. The agent "
    "explains what the tools return; it never approves payment or "
    "decides a claim. Claim submission needs your explicit "
    "confirmation and is idempotent (same key twice = one claim).</div>"
)

# ---------------------------------------------------------------------------
# Agent lifecycle: one long-lived agent (and MCP server subprocess) per
# process, built lazily on the first question. Turns are serialized —
# the Strands loop and its MCP client are not built for concurrent turns.
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: dict = {"cm": None, "agent": None, "live": False, "history": []}


def _desired_live() -> bool:
    load_dotenv()
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _build_agent(live: bool):
    """Enter the repo's eligibility_agent context manager and keep it."""
    if live:
        cm = eligibility_agent(provider="gemini")
    else:
        cm = eligibility_agent(model=ScriptedModel())
    agent = cm.__enter__()
    return cm, agent


def _ensure_agent():
    if _state["agent"] is None:
        live = _desired_live()
        cm, agent = _build_agent(live)
        _state.update(cm=cm, agent=agent, live=live)
    return _state["agent"]


def _rebuild_scripted():
    """Drop whatever agent we have and start fresh on the scripted model."""
    cm = _state.get("cm")
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
    cm, agent = _build_agent(live=False)
    _state.update(cm=cm, agent=agent, live=False)
    return agent


def _mode_label() -> str:
    if _state["agent"] is not None:
        live = _state["live"]
    else:
        live = _desired_live()
    return (
        "live mode (Gemini gemini-3.5-flash-lite)"
        if live
        else "scripted mock mode (deterministic, no key)"
    )


# ---------------------------------------------------------------------------
# One chat turn: run the agent, pull the tool calls + results out of the
# new messages, and detect a submit_claim waiting on confirmation.
# ---------------------------------------------------------------------------


def _parse_tool_input(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _extract_tool_calls(messages) -> list[dict]:
    """Pair toolUse blocks with their toolResult blocks, in call order."""
    calls: list[dict] = []
    by_id: dict[str, dict] = {}
    for msg in messages:
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                use = block["toolUse"]
                entry = {
                    "name": use.get("name", ""),
                    "input": _parse_tool_input(use.get("input")),
                    "result": None,
                }
                calls.append(entry)
                by_id[use.get("toolUseId")] = entry
            elif "toolResult" in block:
                result = block["toolResult"]
                entry = by_id.get(result.get("toolUseId"))
                text = "".join(
                    c.get("text", "")
                    for c in result.get("content", [])
                    if isinstance(c, dict)
                )
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    parsed = text
                if entry is not None:
                    entry["result"] = parsed
    return calls


def _pending_confirmation(calls: list[dict]) -> dict | None:
    """A submit_claim the tool refused for lack of confirmation."""
    for call in calls:
        result = call.get("result")
        if (
            call["name"] == "submit_claim"
            and isinstance(result, dict)
            and not result.get("submitted")
            and "Confirmation required" in str(result.get("reason", ""))
        ):
            return dict(call["input"])
    return None


def _run_turn(prompt: str) -> dict:
    """Run one question through the agent; never raises to the page."""
    with _lock:
        agent = _ensure_agent()
        before = len(agent.messages)
        note = None
        try:
            result = agent(prompt)
        except Exception:
            if not _state["live"]:
                raise
            # Live call failed (quota, network, bad key…): rebuild on
            # the scripted model and answer the same question, so the
            # demo never dies on a network hiccup.
            agent = _rebuild_scripted()
            before = len(agent.messages)
            result = agent(prompt)
            note = (
                "Live mode (Gemini) failed, so this answer came from "
                "the scripted mock model instead."
            )
        calls = _extract_tool_calls(agent.messages[before:])
        turn = {
            "user": prompt,
            "agent": result_text(result),
            "tools": calls,
            "note": note,
            "pending": _pending_confirmation(calls),
        }
        _state["history"].append(turn)
        del _state["history"][:-MAX_TURNS]
        return turn


def _confirm_prompt(details: dict) -> str:
    """The confirmed follow-up, worded like demo scenario 5a."""
    amount = details.get("billed_amount") or 0
    text = (
        f"Submit a claim for member {details.get('member_id', '')}, "
        f"procedure {details.get('procedure_code', '')}, "
        f"billed ${float(amount):.2f}"
    )
    if details.get("provider_npi"):
        text += f", provider {details['provider_npi']}"
    text += (
        f", idempotency key {details.get('idempotency_key', 'web-confirm')}. "
        "I confirm this submission."
    )
    return text


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _page(title, body):
    return HTMLResponse(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)} — agent-eligibility-claims</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n"
        "<header><h1><a href=\"/\">agent-eligibility-claims</a></h1>"
        "<p>Coverage, claim status, and patient cost — decided by "
        "deterministic MCP tools, explained by the agent. Mock data, "
        "no real PHI.</p></header>\n"
        f"<main>{body}</main>\n</body>\n</html>"
    )


def _fmt_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f'{key}="{value}"')
        elif isinstance(value, bool):
            parts.append(f"{key}={'true' if value else 'false'}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _digest(name: str, result) -> str:
    """One readable line of the result's key fields; full JSON is below."""
    if not isinstance(result, dict):
        return html.escape(str(result))[:300]
    r = result
    if name == "verify_eligibility":
        if r.get("eligible"):
            return (
                f"eligible · {r.get('plan')} · deductible remaining "
                f"${r.get('deductible_remaining')} · copay "
                f"${r.get('copay_office_visit')} · prior auth required: "
                f"{'yes' if r.get('prior_auth_required') else 'no'}"
            )
        return f"not eligible · {r.get('reason', '')}"
    if name == "check_claim_status":
        if not r.get("found"):
            return str(r.get("reason", "not found"))
        text = f"status: {r.get('status')} · billed ${r.get('billed_amount')}"
        if r.get("status") == "paid":
            text += (
                f" · allowed ${r.get('allowed_amount')} · paid "
                f"${r.get('paid_amount')} · patient owes "
                f"${r.get('patient_responsibility')}"
            )
        if r.get("status") == "denied":
            text += f" · denial: {r.get('denial_reason')}"
        return text
    if name == "estimate_patient_cost":
        if not r.get("estimable"):
            return f"not estimable · {r.get('reason', '')}"
        return (
            f"patient pays ${r.get('patient_total')} (allowed "
            f"${r.get('allowed_amount')} of ${r.get('billed_amount')} billed; "
            f"deductible ${r.get('deductible_applied')}, coinsurance "
            f"${r.get('coinsurance_owed')}, copay ${r.get('copay')}) · "
            f"plan pays ${r.get('plan_pays')}"
        )
    if name == "submit_claim":
        if r.get("submitted") and r.get("duplicate"):
            return (
                f"already submitted (idempotency key reused) · original "
                f"claim {r.get('claim_id')}, status {r.get('status')}"
            )
        if r.get("submitted"):
            return f"submitted · claim {r.get('claim_id')}, status {r.get('status')}"
        return f"not submitted · {r.get('reason', '')}"
    return ", ".join(f"{k}={v}" for k, v in list(r.items())[:4])


def _tool_call_html(call: dict) -> str:
    result = call.get("result")
    pretty = (
        json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
    )
    return (
        '<div class="toolcall">'
        f'<div class="call">→ {html.escape(call["name"])}'
        f'({html.escape(_fmt_args(call["input"]))})</div>'
        f'<div class="digest">{html.escape(_digest(call["name"], result))}</div>'
        "<details><summary>full tool result (JSON)</summary>"
        f'<pre class="output">{html.escape(pretty)}</pre></details>'
        "</div>"
    )


def _turn_html(turn: dict) -> str:
    tools_html = "".join(_tool_call_html(c) for c in turn["tools"])
    note_html = (
        f'<div class="warn" style="margin-top:6px">{html.escape(turn["note"])}</div>'
        if turn.get("note")
        else ""
    )
    return (
        '<div class="turn">'
        '<div class="who">You</div>'
        f'<div class="user">{html.escape(turn["user"])}</div>'
        '<div class="who" style="margin-top:6px">Agent</div>'
        f'<div class="agent">{html.escape(turn["agent"])}</div>'
        f"{tools_html}{note_html}</div>"
    )


def _confirm_panel(details: dict) -> str:
    amount = details.get("billed_amount") or 0
    hidden = "".join(
        f'<input type="hidden" name="{html.escape(k)}" '
        f'value="{html.escape(str(v))}">'
        for k, v in details.items()
    )
    return (
        '<section class="panel confirm-panel"><h2>Confirm before '
        "submitting</h2>"
        '<p class="meta" style="margin-top:0">The agent tried to file '
        "this claim and the <code>submit_claim</code> tool refused — "
        "submission is a commitment, so it needs a human yes first. "
        "Nothing has been submitted yet.</p>"
        "<table>"
        f"<tr><th>Member</th><td>{html.escape(str(details.get('member_id', '')))}</td></tr>"
        f"<tr><th>Procedure</th><td>{html.escape(str(details.get('procedure_code', '')))}</td></tr>"
        f"<tr><th>Billed amount</th><td>${float(amount):.2f}</td></tr>"
        f"<tr><th>Provider</th><td>{html.escape(str(details.get('provider_npi') or '—'))}</td></tr>"
        f"<tr><th>Idempotency key</th><td><code>{html.escape(str(details.get('idempotency_key', '')))}</code></td></tr>"
        "</table>"
        '<div class="btnrow">'
        '<form method="post" action="/confirm">'
        f"{hidden}"
        '<button type="submit" class="run-big" style="margin-top:0">'
        "Yes — submit this claim</button></form>"
        '<form method="get" action="/">'
        '<button type="submit" class="altbtn" style="margin-top:0">'
        "Cancel — don't submit</button></form>"
        "</div>"
        '<p class="hint">Submitting twice with the same idempotency '
        "key is safe: the second call returns the original claim "
        "instead of creating a duplicate. Try it.</p></section>"
    )


def _mock_data_section() -> str:
    member_rows = []
    for m in logic.MEMBERS.values():
        remaining = round(m["deductible_total"] - m["deductible_met"], 2)
        member_rows.append(
            "<tr>"
            f"<td><code>{html.escape(m['member_id'])}</code></td>"
            f"<td>{html.escape(m['name'])}</td>"
            f"<td>{html.escape(m['plan'])}</td>"
            f"<td>{html.escape(m['coverage_status'])}</td>"
            f"<td>${remaining:.2f}</td>"
            f"<td>${m['copay_office_visit']:.2f}</td>"
            "</tr>"
        )
    procedure_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(code)}</code></td>"
        f"<td>{html.escape(p['description'])}</td>"
        f"<td>{'yes' if p['requires_prior_auth'] else 'no'}</td>"
        "</tr>"
        for code, p in sorted(logic.PROCEDURES.items())
    )
    claim_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(c['claim_id'])}</code></td>"
        f"<td><code>{html.escape(c['member_id'])}</code></td>"
        f"<td><code>{html.escape(c['procedure_code'])}</code></td>"
        f"<td>${c['billed_amount']:.2f}</td>"
        f"<td>{html.escape(c['status'])}</td>"
        "</tr>"
        for c in logic.load_seed_claims()
    )
    return (
        '<section class="panel"><h2>The mock payer data</h2>'
        "<table><tr><th>Member</th><th>Name</th><th>Plan</th>"
        "<th>Coverage</th><th>Deductible left</th><th>Copay</th></tr>"
        f"{''.join(member_rows)}</table>"
        "<table style=\"margin-top:10px\"><tr><th>Procedure</th>"
        "<th>Description</th><th>Prior auth</th></tr>"
        f"{procedure_rows}</table>"
        "<table style=\"margin-top:10px\"><tr><th>Seed claim</th>"
        "<th>Member</th><th>Procedure</th><th>Billed</th><th>Status</th></tr>"
        f"{claim_rows}</table>"
        '<p class="hint">All synthetic. Claims you submit here live in '
        "the agent's in-memory ledger for this session (first one is "
        "CLM-9101) — look them up by ID, or Reset to clear them. "
        "In-network providers: NPI-1111111111 / NPI-2222222222 "
        "(MBR-1004: NPI-3333333333 only).</p></section>"
    )


def _system_prompt_section() -> str:
    return (
        '<details class="panel"><summary>System prompt</summary>'
        '<p class="meta" style="margin-top:10px">Verbatim from '
        "<code>agent.py</code>. The agent writes message text only — "
        "every coverage, cost, and claim decision is made by the "
        "deterministic tools. In scripted mock mode the stand-in "
        "model follows the same rules (one tool call per question, "
        "confirmation before submitting):</p>"
        f'<pre class="output">{html.escape(SYSTEM_PROMPT)}</pre>'
        "</details>"
    )


def _home_page(error: str | None = None, pending: dict | None = None):
    samples = "".join(
        '<form method="post" action="/chat">'
        f'<input type="hidden" name="message" value="{html.escape(q, quote=True)}">'
        f'<button type="submit">{html.escape(label)}</button></form>'
        for label, q in SAMPLE_QUESTIONS
    )
    history_html = ""
    if _state["history"]:
        history_html = (
            '<section class="panel"><h2>Conversation</h2>'
            + "".join(_turn_html(t) for t in _state["history"])
            + '<div class="resetrow"><form method="post" action="/reset">'
            '<button type="submit">Reset conversation + claim ledger</button>'
            "</form></div></section>"
        )
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    confirm_html = _confirm_panel(pending) if pending else ""
    body = (
        '<p class="meta" style="margin-top:0">A patient asks '
        "&ldquo;is this covered, and what will I pay?&rdquo; — and the "
        "front desk burns 20 minutes on hold with the payer. This "
        "agent answers in seconds: coverage, claim status, and the "
        "patient's cost, decided by deterministic tools and explained "
        "in plain language. Ask it something.</p>"
        # The chat panel is the first thing on the page: nothing
        # runs until the visitor asks a question.
        '<section class="panel run-panel"><h2>Ask the agent</h2>'
        f'<p class="meta" style="margin-top:0">Mode: '
        f'<span class="badge">{html.escape(_mode_label())}</span></p>'
        '<form method="post" action="/chat">'
        '<textarea name="message" '
        'placeholder="Ask: Is member MBR-1001 covered for procedure 99213? '
        'What would they pay?"></textarea>'
        '<button type="submit" class="run-big">Ask</button></form>'
        f'<div class="samples">{samples}</div>'
        '<p class="hint">The scripted mock model routes by keywords + '
        "IDs: say <em>eligibility</em>, <em>status</em>, <em>cost</em>, "
        "or <em>submit</em> with a member ID (MBR-1001…1004), a "
        "5-digit procedure code, or a claim ID (CLM-9001…9004). "
        "Live mode (Gemini) understands free-form phrasing.</p>"
        "</section>"
        f"{error_html}{confirm_html}{history_html}{DISCLAIMER}"
        f"{_mock_data_section()}"
        f"{_system_prompt_section()}"
        '<section class="panel"><h2>Run it yourself</h2>'
        f'<p class="meta">Code: <a href="{GITHUB_URL}">agent-eligibility-claims</a> '
        "on GitHub.<br>"
        "This UI locally: <code>pip install -r web/requirements.txt</code> "
        "then <code>uvicorn web.app:app --reload</code> "
        "(or <code>make web</code>).<br>"
        "CLI demo: <code>make demo-mock</code> (offline) or "
        "<code>make demo-gemini</code> (live, free tier). "
        "Tests: <code>make test</code>.</p></section>"
    )
    return _page("Ask the agent", body)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return _home_page()


@app.post("/chat", response_class=HTMLResponse)
def chat(message: str = Form(default="")):
    prompt = (message or "").strip()
    if not prompt:
        return _home_page(
            error="Type a question first — or click one of the samples."
        )
    try:
        turn = _run_turn(prompt)
    except Exception as exc:  # show it, never 500
        return _home_page(error=f"The agent hit a snag: {exc}")
    return _home_page(pending=turn.get("pending"))


@app.post("/confirm", response_class=HTMLResponse)
def confirm(
    member_id: str = Form(default=""),
    procedure_code: str = Form(default=""),
    billed_amount: str = Form(default="0"),
    provider_npi: str = Form(default=""),
    idempotency_key: str = Form(default=""),
):
    details = {
        "member_id": member_id,
        "procedure_code": procedure_code,
        "billed_amount": billed_amount,
        "provider_npi": provider_npi,
        "idempotency_key": idempotency_key,
    }
    try:
        _run_turn(_confirm_prompt(details))
    except Exception as exc:
        return _home_page(error=f"The agent hit a snag: {exc}")
    return _home_page()


@app.post("/reset", response_class=HTMLResponse)
def reset():
    with _lock:
        cm = _state.get("cm")
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
        _state.update(cm=None, agent=None, live=False, history=[])
    return _home_page()
