from pathlib import Path
from string import Template


def build(context):
    if context["plugins"].get("strict_token_retry") is not True:
        return ""
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    return Template(text).substitute(allowed_tokens=" ".join(context["allowed_tokens"])).strip()
