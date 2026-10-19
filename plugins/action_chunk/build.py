from pathlib import Path
from string import Template
import re

from core.action_units import MOVE_ATOMS
from core.step_size import has_explicit_size


def build(context):
    if context["plugins"].get("action_chunk") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    step_num = context["harness"]["action_chunk_steps"]
    return Template(text).substitute(step_num=step_num).strip()


def read_response(context):
    """Save only valid translation plans when the wrist judgment says the target is far."""
    response = context["response"]
    answer = response.payload.get("json", {})
    plan = []
    if has_explicit_size(answer) or "json_error" in response.payload:
        clear(context)
    elif context["plugins"].get("action_chunk") is True and context.get("target_in_wrist") is False:
        text = answer.get("reasoning", response.raw_text) or ""
        marker = re.search(r"PLAN\s*[:=]\s*(.+)", text, re.IGNORECASE)
        if marker:
            moves = re.findall(r"MV_[A-Z]+", marker.group(1).upper())
            plan = [move for move in moves if move in MOVE_ATOMS][:context["harness"]["action_chunk_steps"]]
    response.payload["chunk_plan"] = plan
    context["chunk_plan"] = plan


def queue_plan(context):
    plan = context.get("chunk_plan", [])
    queue = context["plugin_state"].setdefault("action_chunk", [])
    queue.clear()
    response = context.get("response")
    if response is not None and (has_explicit_size(response.payload.get("json", {})) or
                                 "json_error" in response.payload):
        return
    if context["plugins"].get("action_chunk") is True and plan and plan[0] == context["token"]:
        queue.extend(plan[1:])


def next_action(context):
    queue = context["plugin_state"].setdefault("action_chunk", [])
    if context["plugins"].get("action_chunk") is not True:
        queue.clear()
    return queue.pop(0) if queue else None


def clear(context):
    context["plugin_state"].setdefault("action_chunk", []).clear()
