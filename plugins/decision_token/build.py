from pathlib import Path
from string import Template

from core.prompting.content import join_content


def build(context):
    if context["plugins"].get("decision_token") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8").strip()
    before, after = text.split("${previous_prompt}")
    before = Template(before).substitute(allowed_tokens=" ".join(context["allowed_tokens"]))
    after = Template(after).substitute(allowed_tokens=" ".join(context["allowed_tokens"]))
    return join_content((before, context["previous_prompt"], after), separator="")
