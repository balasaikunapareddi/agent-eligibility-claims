"""A deterministic stand-in for a real LLM, for tests and the offline demo.

The scripted model reads the conversation, picks the one tool call the
user's request needs (from keywords and the IDs in the message), and after
the tool result comes back it writes a plain-English summary of what the
tool returned. No network, no API key, same output every run.

It speaks the same stream-event dialect as the real Strands model providers
(``messageStart`` / ``contentBlockStart`` / ``contentBlockDelta`` /
``contentBlockStop`` / ``messageStop`` / ``metadata``), so the agent loop
treats it exactly like a live model.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

_MEMBER_RE = re.compile(r"MBR-\d+", re.IGNORECASE)
_CLAIM_RE = re.compile(r"CLM-\d+", re.IGNORECASE)
_PROCEDURE_RE = re.compile(r"\b(\d{5})\b")
_BILLED_RE = re.compile(r"billed\s+\$?([\d,]+(?:\.\d{2})?)", re.IGNORECASE)
_PROVIDER_RE = re.compile(r"NPI-\d+", re.IGNORECASE)
_KEY_RE = re.compile(r"idempotency key\s+([A-Za-z0-9\-_]+)", re.IGNORECASE)


def _last_user_text(messages: Messages) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            parts = [
                b.get("text", "")
                for b in msg.get("content", [])
                if isinstance(b, dict) and "text" in b
            ]
            if parts:
                return " ".join(parts)
    return ""


def _last_tool_use(messages: Messages) -> tuple[str, dict[str, Any]] | None:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and "toolUse" in block:
                use = block["toolUse"]
                return use.get("name", ""), use.get("input", {}) or {}
    return None


def _pending_tool_result(messages: Messages) -> dict[str, Any] | str | None:
    """The tool result awaiting a summary, or None.

    Only the LAST message counts: a tool result is pending when the most
    recent message is a user message carrying a toolResult. Anything older
    was already summarized in a previous turn — re-summarizing it would
    answer the new request with the old result and skip the new tool call.
    """
    if not messages:
        return None
    last = messages[-1]
    if last.get("role") != "user":
        return None
    for block in last.get("content", []):
        if isinstance(block, dict) and "toolResult" in block:
            result = block["toolResult"]
            texts = [
                c.get("text", "")
                for c in result.get("content", [])
                if isinstance(c, dict) and "text" in c
            ]
            raw = "".join(texts)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
    return None


class ScriptedModel(Model):
    """Rule-based model: keywords + IDs in, one tool call out, then a summary."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {"model_id": "scripted"}

    def get_config(self) -> Any:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:  # type: ignore[override]
        raise NotImplementedError("The scripted model does not support structured output.")

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        result = _pending_tool_result(messages)
        if result is not None:
            tool_use = _last_tool_use(messages)
            name = tool_use[0] if tool_use else ""
            text = self._summarize(name, result)
            async for event in self._text_turn(text):
                yield event
            return

        call = self._decide_tool_call(_last_user_text(messages))
        if call is None:
            text = (
                "I can check eligibility, look up a claim, estimate a patient's "
                "cost, or submit a claim. Give me a member ID (like MBR-1001) "
                "and a procedure code (like 99213), or a claim ID (like CLM-9001)."
            )
            async for event in self._text_turn(text):
                yield event
            return

        name, args = call
        async for event in self._tool_turn(name, args):
            yield event

    # -- turn shapes -----------------------------------------------------

    async def _text_turn(self, text: str) -> AsyncGenerator[StreamEvent, None]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield self._metadata()

    async def _tool_turn(
        self, name: str, args: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "name": name,
                        "toolUseId": f"scripted-{uuid.uuid4().hex[:8]}",
                    }
                }
            }
        }
        yield {
            "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}
        }
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield self._metadata()

    @staticmethod
    def _metadata() -> StreamEvent:
        return {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }

    # -- deciding ----------------------------------------------------------

    def _decide_tool_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        member = _MEMBER_RE.search(text)
        claim = _CLAIM_RE.search(text)
        procedure = _PROCEDURE_RE.search(text)
        provider = _PROVIDER_RE.search(text)
        billed = _BILLED_RE.search(text)
        key = _KEY_RE.search(text)
        lowered = text.lower()

        if "submit" in lowered and member and procedure:
            return "submit_claim", {
                "member_id": member.group(0).upper(),
                "procedure_code": procedure.group(1),
                "billed_amount": float(billed.group(1).replace(",", ""))
                if billed
                else 0.0,
                "provider_npi": provider.group(0).upper() if provider else "",
                "idempotency_key": key.group(1) if key else "scripted-key",
                "confirmed": "confirm" in lowered,
            }
        if claim and ("status" in lowered or "claim" in lowered):
            return "check_claim_status", {"claim_id": claim.group(0).upper()}
        if ("cost" in lowered or "estimate" in lowered or "pay" in lowered) and member and procedure:
            return "estimate_patient_cost", {
                "member_id": member.group(0).upper(),
                "procedure_code": procedure.group(1),
                "billed_amount": float(billed.group(1).replace(",", ""))
                if billed
                else 0.0,
            }
        if ("eligib" in lowered or "cover" in lowered) and member and procedure:
            return "verify_eligibility", {
                "member_id": member.group(0).upper(),
                "procedure_code": procedure.group(1),
                "provider_npi": provider.group(0).upper() if provider else "",
            }
        return None

    # -- summarizing --------------------------------------------------------

    def _summarize(self, tool_name: str, result: dict[str, Any] | str) -> str:
        if not isinstance(result, dict):
            return str(result)
        if tool_name == "verify_eligibility":
            if result.get("eligible"):
                return (
                    f"Member {result.get('member_id')} is eligible for "
                    f"{result.get('procedure_code')} "
                    f"({result.get('procedure_description')}): active "
                    f"{result.get('plan')}, in-network, "
                    f"${result.get('deductible_remaining')} deductible remaining, "
                    f"${result.get('copay_office_visit')} copay. Prior authorization "
                    f"required: {'yes' if result.get('prior_auth_required') else 'no'}."
                )
            return (
                f"Member {result.get('member_id')} is NOT eligible for "
                f"{result.get('procedure_code')}: {result.get('reason')}"
            )
        if tool_name == "check_claim_status":
            if not result.get("found"):
                return result.get("reason", "Claim not found.")
            status = result.get("status")
            text = f"Claim {result.get('claim_id')} is {status}."
            if result.get("billed_amount") is not None:
                text += f" Billed ${result.get('billed_amount')}."
            if status == "paid":
                text += (
                    f" Allowed ${result.get('allowed_amount')}, paid "
                    f"${result.get('paid_amount')}, patient owes "
                    f"${result.get('patient_responsibility')}."
                )
            if status == "denied":
                text += f" Denial reason: {result.get('denial_reason')}"
            if result.get("notes"):
                text += f" {result.get('notes')}"
            return text
        if tool_name == "estimate_patient_cost":
            if not result.get("estimable"):
                return f"I can't estimate that: {result.get('reason')}"
            return (
                f"Estimated patient cost for {result.get('procedure_code')}: "
                f"${result.get('patient_total')} (allowed "
                f"${result.get('allowed_amount')} on ${result.get('billed_amount')} "
                f"billed; deductible ${result.get('deductible_applied')}, "
                f"coinsurance ${result.get('coinsurance_owed')}, copay "
                f"${result.get('copay')}). The plan pays "
                f"${result.get('plan_pays')}."
            )
        if tool_name == "submit_claim":
            if result.get("submitted") and result.get("duplicate"):
                return (
                    "Already submitted — this idempotency key was used before, "
                    "so no new claim was created. Original claim: "
                    f"{result.get('claim_id')} (status {result.get('status')})."
                )
            if result.get("submitted"):
                return (
                    f"Claim submitted: {result.get('claim_id')}, status "
                    f"{result.get('status')}."
                )
            return f"I didn't submit the claim: {result.get('reason')}"
        return json.dumps(result)
