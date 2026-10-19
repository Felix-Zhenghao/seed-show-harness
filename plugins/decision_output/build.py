from pathlib import Path
from string import Template

from core.step_size import size_bounds

def build(context):
    if context["plugins"].get("decision_output") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    actions = context["semantic_eef_action"]
    distance_low, distance_high = size_bounds(actions, "distance_m")
    angle_low, angle_high = size_bounds(actions, "angle_deg")
    return Template(text).substitute(
        allowed_tokens=", ".join(context["allowed_tokens"]),
        distance_bounds=f"[{distance_low:g}, {distance_high:g}]",
        angle_bounds=f"[{angle_low:g}, {angle_high:g}]",
    ).strip()
