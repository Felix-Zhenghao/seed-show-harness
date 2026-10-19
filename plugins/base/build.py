"""Fill the base prompt's live values; all instructions are written in prompt.txt."""
import json
from pathlib import Path
from string import Template

from core.prompting.content import join_content


def build(context):
    return _build(context)


def build_instructions(context):
    """Rules that depend only on the configured action space and step sizes."""
    return _build(context, "instructions")


def build_state(context):
    """Task and subgoal data, kept after reusable rules by the composer."""
    return _build(context, "state")


def _build(context, section=None):
    if context["plugins"].get("base") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    instructions, _, state = text.partition("\n[state]\n")
    if section == "instructions":
        text = instructions
    elif section == "state":
        text = state
    else:
        text = join_content((instructions.strip(), state.strip()))
    actions = context["semantic_eef_action"]
    stage = dict(sorted((context.get("subgoal") or {}).items()))
    return Template(text).substitute(
        allowed_tokens=", ".join(context["allowed_tokens"]),
        step_cm=f"{actions['step_m'] * 100:g}",
        rotation_deg=f"{actions['rotation_step_deg']:g}",
        affordance=stage.get("affordance") or "the intended grasping part",
        task=context["task"],
        subgoal=json.dumps({**stage, "gripper_state": context["gripper_state"]},
                           indent=2, ensure_ascii=False),
    ).strip()
