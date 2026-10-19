import json
from pathlib import Path
from string import Template


def build(context):
    if context["plugins"].get("proprioception") is not True:
        return ""
    state = context.get("proprioception")
    if not state:
        return ""
    height = ""
    eef, table = state.get("eef_pos", []), context.get("table_height_m")
    if len(eef) >= 3 and table is not None:
        height = Template(_text("height")).substitute(gap_cm=f"{(eef[2] - table) * 100:.1f}")
    return Template(_text()).substitute(
        state=json.dumps(state), height=height,
    )


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()
