from pathlib import Path


def build(context):
    if context["plugins"].get("decision_json") is not True:
        return ""
    return Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8").strip()
