"""Detect failed or unsettled grasps and request holds, reopening, or plan rewind."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Sequence

RELEASE_TOKEN = "RELEASE"
GRASP_TOKEN = "GRASP"
STOP_TOKEN = "STOP"

# Bound the wait for a newly closed gripper to leave the open-width band.
MAX_UNSETTLED_HOLDS = 3


@dataclass(frozen=True)
class RecoveryDecision:
    """A runner-level recovery request.

    ``token`` is an optional action override for the current step. ``release`` requests
    an immediate release after the just-executed token. ``rollback_index`` moves the
    active subgoal pointer before the next VLM decision.
    """

    event: str
    reason: str
    token: str | None = None
    release: bool = False
    rollback_index: int | None = None
    block_done: bool = False
    reset_history: bool = False
    grasp_empty: bool = False
    grasp_unverified: bool = False


def build(context):
    if context["plugins"].get("recovery") is not True:
        return ""
    notice = context.get("recovery")
    if notice is None or isinstance(notice, str):
        return ""
    event = notice.event
    if event == "unverified_grasp" and notice.grasp_empty:
        event = "empty_grasp"
    return Template(_text()).substitute(note=_text(event))


def phase_from_width(context, width_m: Any) -> str:
    """Classify measured width as ``empty``, ``holding``, ``open``, or ``unknown``."""
    try:
        width = float(width_m)
    except (TypeError, ValueError):
        return "unknown"
    if width <= context['harness']['empty_width_m']:
        return "empty"
    if width >= context['harness']['open_width_m']:
        return "open"
    return "holding"


def before_decision(context) -> RecoveryDecision | None:
    """Reopen empty grasps or briefly hold while a close settles."""
    if context["plugins"].get("recovery") is not True:
        return None
    current_index, subgoals = context["stage_index"], context["subgoals"]
    measured_width_m = context["proprioception"]["gripper_width_m"]
    gripper_closed = context["gripper_state"] == "CLOSED"
    state = context["plugin_state"].setdefault("recovery", {"unsettled_holds": 0})
    if not gripper_closed:
        state["unsettled_holds"] = 0
        return None
    phase = phase_from_width(context, measured_width_m)

    if phase == "empty":
        state["unsettled_holds"] = 0
        if not context["plugins"].get("auto_release", False):
            return None
        rollback = _nearest_grasp_index(subgoals, current_index)
        if rollback is None:
            return None
        stage = (
            _motion(subgoals[current_index])
            if 0 <= current_index < len(subgoals)
            else ""
        )
        event = "empty_grasp" if stage == GRASP_TOKEN else "lost_grasp"
        reason = (
            f"closed gripper width {float(measured_width_m):.4f}m <= "
            f"empty threshold {context['harness']['empty_width_m']:.4f}m"
        )
        return RecoveryDecision(
            event=event,
            reason=reason,
            token=RELEASE_TOKEN,
            rollback_index=rollback,
            block_done=True,
            reset_history=True,
            grasp_empty=True,
            grasp_unverified=True,
        )

    if phase != "open":  # holding or unknown -> proceed normally
        state["unsettled_holds"] = 0
        return None

    # A bounded hold prevents movement before the close has settled.
    if state["unsettled_holds"] >= MAX_UNSETTLED_HOLDS:
        state["unsettled_holds"] = 0
        return None
    state["unsettled_holds"] += 1
    return RecoveryDecision(
        event="grasp_unsettled",
        reason=(
            f"closed gripper still reads open width {_fmt_width(measured_width_m)}; "
            "waiting for the gripper sensor to settle before the next move"
        ),
        token=STOP_TOKEN,
        block_done=True,
    )


def after_step(context) -> RecoveryDecision | None:
    """Validate a just-executed step before the runner advances subgoals."""
    if context["plugins"].get("recovery") is not True:
        return None
    current_index, subgoals = context["stage_index"], context["subgoals"]
    measured_width_m = context["proprioception"]["gripper_width_m"]
    gripper_closed = context["gripper_state"] == "CLOSED"
    subgoal_done = context["subgoal_done"]
    rollback = _nearest_grasp_index(subgoals, current_index)
    if rollback is None:
        return None

    stage = (
        _motion(subgoals[current_index])
        if 0 <= current_index < len(subgoals)
        else ""
    )
    phase = phase_from_width(context, measured_width_m)

    if stage == GRASP_TOKEN and subgoal_done and phase != "holding":
        reason = (
            f"grasp DONE rejected because measured width phase is {phase}"
            f" (width={_fmt_width(measured_width_m)})"
        )
        return RecoveryDecision(
            event="unverified_grasp",
            reason=reason,
            release=bool(context["plugins"].get("auto_release", False) and gripper_closed and phase == "empty"),
            rollback_index=current_index,
            block_done=True,
            reset_history=True,
            grasp_empty=(phase == "empty"),
            grasp_unverified=True,
        )

    if stage != GRASP_TOKEN and gripper_closed and phase == "empty":
        reason = (
            f"closed gripper became empty during {stage or 'post-grasp stage'} "
            f"(width={_fmt_width(measured_width_m)})"
        )
        return RecoveryDecision(
            event="lost_grasp",
            reason=reason,
            release=context["plugins"].get("auto_release", False),
            rollback_index=rollback,
            block_done=True,
            reset_history=True,
            grasp_empty=True,
            grasp_unverified=True,
        )

    return None


def _nearest_grasp_index(subgoals: Sequence[Any], current_index: int) -> int | None:
    upper = min(max(int(current_index), 0), len(subgoals) - 1)
    for idx in range(upper, -1, -1):
        if _motion(subgoals[idx]) == GRASP_TOKEN:
            return idx
    return 0 if subgoals else None


def _motion(subgoal: Any) -> str:
    return str(getattr(subgoal, "motion", "") or "").strip().upper()


def _fmt_width(value: Any) -> str:
    try:
        return f"{float(value):.4f}m"
    except (TypeError, ValueError):
        return "unknown"


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()
