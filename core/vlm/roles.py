from __future__ import annotations

import json
from typing import Any, Sequence

from core.action_units import MOVE_ATOMS
from core.prompting.composer import controller_prompt, decision_prompt, token_prompt
from core.prompting.content import Prompt, prompt_text
from core.step_size import parse_step_size, size_bounds
from plugins.action_chunk import build as action_chunk
from plugins.wrist_visibility import build as wrist_visibility

from .vlm_client import VLMParseError, VLMResponse, recover_allowed_token


def _decision_schema(allowed_tokens: Sequence[str], actions: dict | None = None) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(allowed_tokens)},
            "reasoning": {"type": "string"},
            "distance_m": {"type": ["number", "null"], "exclusiveMinimum": 0,
                           "description": "Optional translation magnitude in meters for MV_* only."},
            "angle_deg": {"type": ["number", "null"], "exclusiveMinimum": 0,
                          "description": "Optional rotation magnitude in degrees for RT_* only."},
            "step_size_reasoning": {"type": "string",
                                    "description": "Explain a size override separately from the action choice."},
        },
        "required": ["decision", "reasoning"],
        "additionalProperties": False,
    }
    for field in ("distance_m", "angle_deg"):
        low, high = size_bounds(actions or {}, field)
        schema["properties"][field].update(minimum=low, maximum=high)
    return schema


def _complete_decision_json(
    client: Any,
    prompt: Prompt,
    token_prompt: Prompt,
    allowed_tokens: Sequence[str],
    agentview_image,
    wrist_image=None,
    fallback_token: str | None = None,
    fallback_reason: str = "fallback after invalid decision JSON and token retry",
    debug: bool = False,
    context: dict | None = None,
) -> VLMResponse:
    try:
        response = client.complete_json(
            prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=_decision_schema(allowed_tokens, context["semantic_eef_action"] if context else None),
            max_tokens=None,
            debug=debug,
            context=context,
        )
        payload = response.payload.get("json")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Decision JSON missing object payload: {response.raw_text!r}")
        decision = str(payload.get("decision", "")).strip()
        if decision not in allowed_tokens:
            raise RuntimeError(
                f"Decision JSON has invalid decision {decision!r}; "
                f"allowed tokens are {list(allowed_tokens)}"
            )
        reasoning = str(payload.get("reasoning") or "").strip()
        normalized = {"decision": decision, "reasoning": reasoning}
        for field in ("distance_m", "angle_deg", "step_size_reasoning"):
            if field in payload:
                normalized[field] = payload[field]
        merged_payload = dict(response.payload)
        merged_payload["json"] = normalized
        return VLMResponse(
            token=decision,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=merged_payload,
        )
    except RuntimeError as exc:
        recovered = _recover_decision_from_malformed_json(exc, allowed_tokens)
        if recovered:
            decision, raw_text = recovered
            normalized = {
                "decision": decision,
                "reasoning": (
                    f"Recovered {decision} from malformed VLM JSON output. "
                    f"Raw output: {_one_line(raw_text)}"
                ),
            }
            payload = {
                "json": normalized,
                "recovered_from_malformed_json": True,
                "json_error": str(exc),
                "latency_s": 0.0,
            }
            return VLMResponse(
                token=decision,
                raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
                payload=payload,
            )
        return _token_retry_or_fallback(
            client=client,
            prompt=token_prompt,
            allowed_tokens=allowed_tokens,
            fallback_token=fallback_token,
            fallback_reason=fallback_reason,
            agentview_image=agentview_image,
            wrist_image=wrist_image,
            debug=debug,
            context=context,
            json_error=exc,
            primary_reasoning=_reasoning_from_exception(exc),
        )


def _token_retry_or_fallback(
    client: Any,
    prompt: Prompt,
    allowed_tokens: Sequence[str],
    fallback_token: str | None,
    fallback_reason: str,
    agentview_image,
    wrist_image=None,
    debug: bool = False,
    json_error: RuntimeError | None = None,
    primary_reasoning: str = "",
    context: dict | None = None,
) -> VLMResponse:
    tokens = tuple(allowed_tokens)
    if not tokens:
        raise RuntimeError("No allowed tokens available for VLM fallback")
    fallback = fallback_token if fallback_token in tokens else None
    try:
        retry = client.complete_token(
            prompt,
            tokens,
            agentview_image,
            wrist_image=wrist_image,
            debug=debug,
            context=context,
        )
        # Keep the model's own reasoning from the first pass; the token just came from
        # a strict-retry call. Only fall back to the bare label when no reasoning exists.
        normalized = {
            "decision": retry.token,
            "reasoning": _retry_reasoning(
                primary_reasoning, "strict token retry after invalid JSON"
            ),
        }
        payload: dict[str, Any] = {"json": normalized, "retry": retry.payload}
        if "latency_s" in retry.payload:
            payload["latency_s"] = retry.payload["latency_s"]
        if json_error is not None:
            payload["json_error"] = str(json_error)
        return VLMResponse(
            token=retry.token,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )
    except RuntimeError as retry_exc:
        if fallback is None:
            raise RuntimeError(
                "Invalid decision JSON and strict token retry failed; "
                f"json_error={json_error}; retry_error={retry_exc}"
            ) from retry_exc
        normalized = {
            "decision": fallback,
            "reasoning": _retry_reasoning(primary_reasoning, fallback_reason),
        }
        payload = {
            "json": normalized,
            "fallback": True,
            "json_error": "" if json_error is None else str(json_error),
            "retry_error": str(retry_exc),
            "latency_s": 0.0,
        }
        return VLMResponse(
            token=fallback,
            raw_text=json.dumps(normalized, ensure_ascii=False, sort_keys=True),
            payload=payload,
        )


def _retry_reasoning(primary_reasoning: str, label: str) -> str:
    """The reasoning to log when the token came from a strict retry / commit fallback.
    Prefer the model's own first-pass reasoning, tagging how the token was recovered so
    the provenance is still visible; fall back to the bare label when none exists."""
    primary = (primary_reasoning or "").strip()
    if primary:
        return f"{primary} [{label}]"
    return label


def _reasoning_from_exception(exc: RuntimeError | None) -> str:
    """Best-effort recovery of the model's raw output text from a parse failure, so it
    can still be logged as the reasoning instead of a generic label."""
    raw = getattr(exc, "raw_text", "") or ""
    return _one_line(raw) if raw else ""


def _recover_decision_from_malformed_json(
    exc: RuntimeError, allowed_tokens: Sequence[str]
) -> tuple[str, str] | None:
    raw_text = str(getattr(exc, "raw_text", "") or "")
    if not raw_text and isinstance(exc, VLMParseError):
        raw_text = str(exc.raw_text or "")
    if not raw_text:
        return None
    token = recover_allowed_token(raw_text, allowed_tokens)
    if not token:
        return None
    return token, raw_text


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


class ControllerAgent:
    """Choose one semantic action from the current stage and camera observations."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.last_prompt = ""

    def decide(self, context: dict, debug: bool = False) -> VLMResponse:
        allowed_tokens = context["allowed_tokens"]
        previous_direction = context["previous_direction"]
        # Build world-dependent chunks once; retries keep the exact same observation context.
        base_prompt = controller_prompt(context)
        output_context = {**context, "previous_prompt": base_prompt}
        prompt = decision_prompt(output_context)
        token_only_prompt = token_prompt(output_context)
        self.last_prompt = prompt_text(prompt)
        fallback_token = previous_direction if previous_direction in MOVE_ATOMS else "MV_DOWN"
        response = _complete_decision_json(
            self.client,
            prompt,
            token_only_prompt,
            allowed_tokens,
            context["agentview_image"],
            wrist_image=context["wrist_image"],
            fallback_token=fallback_token,
            fallback_reason="controller fallback after invalid JSON and token retry",
            debug=debug,
            context=context,
        )
        context["response"] = response
        response.payload["step_size"] = parse_step_size(
            response.token, response.payload["json"], context["semantic_eef_action"],
        )
        wrist_visibility.read_response(context)
        action_chunk.read_response(context)
        return response
