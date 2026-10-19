from pathlib import Path
from string import Template

from core.prompting.content import join_content


def build(context):
    return join_content((build_instructions(context), build_state(context)))


def build_instructions(context):
    if context["plugins"].get("mem_text") is not True:
        return ""
    return _text()


def build_state(context):
    if context["plugins"].get("mem_text") is not True:
        return ""
    # The runner records oldest first; prompt order is owned here.
    recent = context["recent_moves"][-context["harness"]["mem_text_len"]:]
    moves = ", ".join(reversed(recent)) if recent else _text("empty")
    return Template(_text("state")).substitute(moves=moves)


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()


def remember(context, executed_tokens):
    if context["plugins"].get("mem_text") is not True:
        return []
    return (context["recent_moves"] + executed_tokens)[-context["harness"]["mem_text_len"]:]
