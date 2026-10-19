"""Reopen an empty closed gripper and describe the intervention."""
from pathlib import Path


def build(context):
    if context["plugins"].get("auto_release") is not True or context.get("recovery") != "auto_release":
        return ""
    return Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8").strip()


def after_step(context, requested=False):
    observation = context["observation"]
    success = context["success"]
    if context["plugins"].get("auto_release") is not True or success:
        return observation, [], success, False
    width = context["proprioception"]["gripper_width_m"]
    empty = context["gripper_state"] == "CLOSED" and width < context["harness"]["empty_width_m"]
    if not (empty or requested):
        return observation, [], success, False
    observation, commands, success = context["executor"].execute("RELEASE", observation)
    return observation, commands, success, True
