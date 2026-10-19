import json
from pathlib import Path
from string import Template


def build(context):
    if context["plugins"].get("whole_task_stage") is not True:
        return ""
    task = context["task"]
    values = {"task": json.dumps(task, ensure_ascii=False)}
    if context.get("fallback"):
        for name in ("description", "completion"):
            values[name] = json.dumps(Template(_text(f"fallback_{name}")).substitute(task=task), ensure_ascii=False)
        return Template(_text("fallback")).substitute(**values)
    return Template(_text()).substitute(**values)


def _text(section="prompt"):
    text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
    if section != "prompt":
        _, marker, text = text.partition(f"\n[{section}]\n")
        if not marker:
            raise ValueError(f"Missing [{section}] in {__file__}'s prompt.txt")
    return text.partition("\n[")[0].strip()
