"""Call the subgoal planner and repair malformed responses without losing context."""
from __future__ import annotations

import json
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from core.types import Subgoal

from core.prompting.content import join_content, prompt_text


@dataclass
class PlannerResponse:
    token: str
    raw_text: str
    payload: dict[str, Any]


# Output contract: an ordered list of subgoals, each fully specified. additionalProperties
# is forbidden and every field required so guided/structured decoding stays strict.
SUBGOAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "target": {"type": "string"},
                    "affordance": {"type": "string"},
                    "motion": {"type": "string"},
                    "description": {"type": "string"},
                    "completion": {"type": "string"},
                },
                "required": [
                    "id",
                    "target",
                    "affordance",
                    "motion",
                    "description",
                    "completion",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}


def build(context):
    if context["plugins"].get("subgoal") is not True:
        return ""
    return _text()


def plan(context):
    """Plan once, repair malformed output, and merge immediate approach/grasp stages."""
    if context["plugins"].get("subgoal") is not True:
        return [], ""
    response = _request_plan(context)
    context["planner_response"] = response
    context["planner_prompt_text"] = prompt_text(context["planner_prompt"])
    payload = response.payload["json"]
    items = _subgoal_items(payload)
    if not items:
        raise RuntimeError(f"Planner JSON has no subgoals: {response.raw_text!r}")
    stages = [Subgoal.from_dict(item, index=i) for i, item in enumerate(items)]
    return _merge_pregrasp_stages(stages), response.raw_text


def _request_plan(context):
    debug = context.get("debug", False)
    started = time.monotonic()
    agentview_image, wrist_image = context["agentview_image"], context["wrist_image"]
    prompt = context["planner_prompt"]
    errors: list[str] = []
    try:
        response = context["client"].complete_json(
            prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=SUBGOAL_PLAN_SCHEMA,
            max_tokens=context["harness"]["planner_max_tokens"],
            debug=debug,
            context=context,
        )
        response = _require_plan_response(response)
        return _finish(context, response, "json", errors, started)
    except RuntimeError as exc:
        if "VLM returned" not in str(exc):
            raise
        errors.append(f"json: {_one_line(str(exc))}")
        # Retry with a stricter prompt and no provider JSON-format constraint.
        strict_prompt = join_content((prompt, _text("json_retry")))
    try:
        response = context["client"].complete_json(
            strict_prompt,
            agentview_image,
            wrist_image=wrist_image,
            schema=None,
            max_tokens=context["harness"]["planner_max_tokens"],
            debug=debug,
            context=context,
        )
        response = _require_plan_response(response)
        return _finish(context, response, "free_json", errors, started)
    except RuntimeError as exc:
        if "VLM returned" not in str(exc):
            raise
        errors.append(f"free_json: {_one_line(str(exc))}")

    if hasattr(context["client"], "complete_text"):
        # Preserve the task and scene context on every parse-repair attempt.
        text_prompt = join_content((prompt, _text("text_retry")))
        for label in ("text_retry_1", "text_retry_2"):
            try:
                response = context["client"].complete_text(
                    text_prompt,
                    agentview_image,
                    wrist_image=wrist_image,
                    max_tokens=context["harness"]["planner_max_tokens"],
                    debug=debug,
                    context=context,
                    strip_reasoning=False,
                )
                parsed = _require_plan_json(_parse_json_object(response.raw_text))
                response = _json_response(parsed, response, source=label, errors=errors)
                return _finish(context, response, label, errors, started)
            except RuntimeError as exc:
                if "VLM returned" not in str(exc):
                    raise
                errors.append(f"{label}: {_one_line(str(exc))}")

    response = _fallback_response(context, errors)
    return _finish(context, response, "task_fallback", errors, started)


def _finish(
    context,
    response: Any,
    route: str,
    errors: "list[str]",
    started: float,
) -> PlannerResponse:
    payload = dict(getattr(response, "payload", {}) or {})
    payload["planner_retry"] = route
    if errors:
        payload["planner_errors"] = list(errors)
    finished = PlannerResponse(
        token=str(getattr(response, "token", "")),
        raw_text=str(getattr(response, "raw_text", "")),
        payload=payload,
    )
    context["planner_diagnostics"] = {
        "route": route,
        "errors": list(errors),
        "latency_s": payload.get("latency_s"),
        "total_latency_s": round(time.monotonic() - started, 3),
        "response_chars": len(finished.raw_text),
        "max_tokens": context["harness"]["planner_max_tokens"],
    }
    return finished


def _parse_json_object(raw_text: Any) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    decoder = json.JSONDecoder()
    for match_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[match_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise RuntimeError(f"VLM returned non-JSON text: {text!r}")


def _require_plan_response(response: Any) -> Any:
    payload = getattr(response, "payload", {}) or {}
    parsed = payload.get("json") if isinstance(payload, dict) else None
    _require_plan_json(parsed)
    return response


def _require_plan_json(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict) and _has_subgoals(parsed):
        return parsed
    raise RuntimeError(
        "VLM returned JSON without non-empty subgoals: "
        f"{json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else parsed!r}"
    )


def _has_subgoals(parsed: dict[str, Any]) -> bool:
    items = parsed.get("subgoals")
    if isinstance(items, list):
        return any(isinstance(item, dict) for item in items)
    return isinstance(items, dict)


def _json_response(
    parsed: dict[str, Any],
    response: Any,
    *,
    source: str,
    errors: list[str],
) -> PlannerResponse:
    payload = dict(getattr(response, "payload", {}) or {})
    payload["json"] = parsed
    payload["planner_retry"] = source
    if errors:
        payload["planner_errors"] = list(errors)
    return PlannerResponse(
        token="",
        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        payload=payload,
    )


def _fallback_response(context: dict, errors: list[str]) -> PlannerResponse:
    if context.get("fallback_stage") is None:
        raise RuntimeError("Planner exhausted retries and whole_task_stage is disabled")
    print(
        "[subgoal] WARNING: planner degraded to a single whole-task stage after "
        f"{len(errors)} failed attempts; first error: {errors[0][:160] if errors else '?'}"
    )
    parsed: dict[str, Any] = {
        "planner_fallback": "subgoal planner VLM returned empty/non-JSON after retries",
        "planner_errors": list(errors),
        "subgoals": [context["fallback_stage"]],
    }
    return PlannerResponse(
        token="",
        raw_text=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
        payload={"json": parsed, "planner_fallback": True},
    )


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def _subgoal_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("subgoals")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    if isinstance(items, dict):
        return [items]
    item = payload.get("subgoal")
    if isinstance(item, dict):
        return [item]
    if _looks_like_subgoal(payload):
        return [payload]
    return []


def _looks_like_subgoal(payload: dict[str, Any]) -> bool:
    required = {"id", "target", "affordance", "motion", "description", "completion"}
    return required.issubset(payload)


def _merge_pregrasp_stages(subgoals: list[Subgoal]) -> list[Subgoal]:
    normalized: list[Subgoal] = []
    i = 0
    while i < len(subgoals):
        current = subgoals[i]
        if i + 1 < len(subgoals):
            next_subgoal = subgoals[i + 1]
            if _should_merge_pregrasp(current, next_subgoal):
                normalized.append(_merge_pregrasp(current, next_subgoal))
                i += 2
                continue
        normalized.append(current)
        i += 1
    return normalized


def _should_merge_pregrasp(current: Subgoal, next_subgoal: Subgoal) -> bool:
    if not _same_target(current.target, next_subgoal.target):
        return False
    if not _is_grasp_stage(next_subgoal):
        return False
    return _is_pure_reach_or_align(current)


def _merge_pregrasp(reach: Subgoal, grasp: Subgoal) -> Subgoal:
    return Subgoal(
        id=grasp.id,
        target=grasp.target or reach.target,
        affordance=grasp.affordance or reach.affordance,
        motion=grasp.motion,
        description=_join_descriptions(reach.description, grasp.description),
        completion=grasp.completion,
    )


def _is_pure_reach_or_align(subgoal: Subgoal) -> bool:
    text = _subgoal_text(subgoal)
    if not _contains_any(
        text,
        (
            "approach",
            "reach",
            "align",
            "center",
            "position",
            "move to",
            "move toward",
            "above",
            "over",
        ),
    ):
        return False
    return not _contains_any(
        text,
        (
            "grasp",
            "close",
            "clamp",
            "secure",
            "lift",
            "release",
            "open",
            "place",
            "drop",
        ),
    )


def _is_grasp_stage(subgoal: Subgoal) -> bool:
    return _contains_any(
        _subgoal_text(subgoal),
        ("grasp", "close", "clamp", "secure", "pick"),
    )


def _same_target(a: str, b: str) -> bool:
    left = _norm_target(a)
    right = _norm_target(b)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _norm_target(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _subgoal_text(subgoal: Subgoal) -> str:
    return " ".join(
        str(getattr(subgoal, field, "") or "").lower()
        for field in ("id", "target", "affordance", "motion", "description", "completion")
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text) for term in terms
    )


def _join_descriptions(first: str, second: str) -> str:
    parts = []
    for text in (first, second):
        stripped = " ".join(str(text or "").split())
        if stripped:
            parts.append(stripped.rstrip("."))
    if not parts:
        return ""
    return ". Then ".join(parts) + "."


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()
