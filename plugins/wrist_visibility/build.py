import re
from typing import Optional

from pathlib import Path


def build(context):
    if context["plugins"].get("wrist_visibility") is not True:
        return ""
    return Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8").strip()


# Explicit marker the VLM writes so we never mistake casual prose for the judgment. The
# trailing \b stops partial-word matches (e.g. "WRIST: NOPE" / "WRIST: YESTERDAY").
_WRIST_MARKER = re.compile(r"WRIST\s*[:=]\s*(YES|NO)\b", re.IGNORECASE)


def parse_wrist_marker(text: str) -> Optional[bool]:
    """Recover the VLM's wrist-visibility judgment from its reasoning/output text.

    Returns True (TARGET in the wrist view), False (not in view), or None when no explicit
    ``WRIST: YES/NO`` marker is present. Uses the LAST marker, i.e. the VLM's final word.
    """
    if not text:
        return None
    matches = _WRIST_MARKER.findall(str(text))
    if not matches:
        return None
    return matches[-1].upper() == "YES"


def read_response(context):
    if context["plugins"].get("wrist_visibility") is not True:
        context["target_in_wrist"] = None
        return
    response = context["response"]
    answer = response.payload.get("json", {})
    text = answer.get("reasoning", response.raw_text) or ""
    context["target_in_wrist"] = parse_wrist_marker(text)
    response.payload["target_in_wrist"] = context["target_in_wrist"]
