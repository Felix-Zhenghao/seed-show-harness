from pathlib import Path
from string import Template


def build(context):
    if context["plugins"].get("extra_context") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    return Template(text).substitute(extra_context=context.get("extra_context", "")).strip()
