"""Describe and choose coarse translations using the current state and wrist judgment."""
from pathlib import Path
from string import Template


def build(context):
    return _build(context)


def build_instructions(context):
    return _build(context, "instructions")


def build_state(context):
    return _build(context, "state")


def _build(context, section=None):
    if context["plugins"].get("variable_step") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    main, hints = text.split("\n[holding]\n")
    holding, descending = hints.split("\n[descending]\n")
    hint = holding if context["gripper_state"] == "CLOSED" else descending
    hint = Template(hint.strip()).substitute(
        high_cm=f"{context['harness']['high_above_table_m'] * 100:g}",
    )
    if section == "state":
        return hint
    return Template(main).substitute(
        coarse_cm=f"{context['harness']['coarse_step_m'] * 100:g}",
        high_cm=f"{context['harness']['high_above_table_m'] * 100:g}",
        hint="" if section == "instructions" else hint,
    ).strip()


def step_m(context):
    fine = context["semantic_eef_action"]["step_m"]
    if context["plugins"].get("variable_step") is not True:
        return fine
    settings = context["harness"]
    if context["token"] == "MV_UP" or context.get("target_in_wrist") is False:
        return settings["coarse_step_m"]
    eef = context["proprioception"].get("eef_pos", [])
    table = context.get("table_height_m")
    if len(eef) >= 3 and table is not None and eef[2] - table > settings["high_above_table_m"]:
        return settings["coarse_step_m"]
    return fine
