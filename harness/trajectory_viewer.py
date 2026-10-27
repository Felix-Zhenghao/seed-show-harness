"""Render saved episode logs as a readable, offline conversation and action history."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from harness.serialization import Object

STYLE = """
:root{color-scheme:light;--ink:#1d2939;--muted:#667085;--line:#dce3ec;--blue:#2458a6}
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:var(--ink);font:15px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:40px auto;padding:0 24px 64px}h1{font-size:32px;margin:4px 0 12px}
h2{font-size:23px;margin:32px 0 16px}h3{font-size:18px;margin:0}h4{font-size:15px;margin:0 0 8px}
p{margin:8px 0 16px}a{color:var(--blue);text-underline-offset:3px}summary{cursor:pointer;font-weight:600}
details>summary{padding:10px 0}details[open]>summary{margin-bottom:10px}
.muted,.meta{color:var(--muted);font-size:13px}.eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase}
.stats{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}.stat{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 24px;min-width:150px}
.stat strong{display:block;font-size:24px}.links{display:flex;flex-wrap:wrap;gap:16px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0}nav a{background:white;border:1px solid var(--line);border-radius:7px;padding:5px 12px;text-decoration:none}
.decision{scroll-margin-top:24px}.event{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:16px 0;min-width:0}
.event-header{display:flex;justify-content:space-between;gap:16px;align-items:baseline;flex-wrap:wrap}
.model-call{border-left:4px solid var(--blue)}.action{border-left:4px solid #33815a}.failure{border-left:4px solid #b34435}
.badge{display:inline-block;border-radius:20px;background:#edf2fa;color:var(--blue);padding:2px 10px;font-size:12px}
.prompt{margin-top:14px;border-top:1px solid var(--line)}.prompt>summary{color:var(--blue)}
.prompt-section{margin:20px 0}.instructions,.configuration{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:4px 16px}
.message{border:1px solid var(--line);border-radius:9px;margin:12px 0;overflow:hidden}
.message-header{background:#f3f6fb;padding:10px 16px;font-weight:650;font-size:13px;display:flex;gap:10px;flex-wrap:wrap}
.message.assistant .message-header{background:#f1edfb}.message.tool .message-header{background:#eef6f2}
.message-body{padding:16px;min-width:0}.prose{white-space:pre-wrap;overflow-wrap:anywhere;margin:0 0 14px}
.message-body>figure{display:inline-block;vertical-align:top;margin:8px 16px 8px 0}
.prose:last-child{margin-bottom:0}.fields{margin:8px 0}.field{display:grid;grid-template-columns:minmax(125px,180px) minmax(0,1fr);gap:16px;padding:9px 0;border-bottom:1px solid #edf0f5}
.field:last-child{border-bottom:0}.field dt{font-size:13px;font-weight:600;color:#475467;overflow-wrap:anywhere}.field dd{margin:0;min-width:0;overflow-wrap:anywhere}
.field .fields{margin:0;border-left:2px solid #edf0f5;padding-left:12px}.field .field{grid-template-columns:minmax(95px,140px) minmax(0,1fr)}
.scalar,.vector,.matrix{font-family:ui-monospace,monospace;font-size:12px}.vector{display:flex;flex-wrap:wrap;gap:5px 12px}.vector span{overflow-wrap:anywhere}
.matrix-wrap{overflow-x:auto;max-width:100%}.matrix{border-collapse:collapse}.matrix td{padding:4px 10px;border:1px solid var(--line);text-align:right}
.items{padding-left:22px;margin:8px 0}.items>li{padding:6px 0}.tool-card{border:1px solid var(--line);border-radius:8px;padding:16px;margin:12px 0}
.tool-name{font-family:ui-monospace,monospace;color:var(--blue)}.schema{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0}
.schema th,.schema td{text-align:left;vertical-align:top;padding:10px;border:1px solid var(--line);overflow-wrap:anywhere}.schema th{background:#f4f6fa}
.schema td:first-child{width:20%}.schema td:nth-child(2){width:18%}.schema .prose{margin-bottom:8px}
.images{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0}figure{margin:0;max-width:100%}figure img{display:block;max-width:100%;width:320px;height:auto;border:1px solid var(--line);border-radius:8px}
figcaption{font-weight:600;font-size:13px;margin:8px 0;color:#475467}.response{margin-top:18px;border-top:1px solid var(--line);padding-top:16px}
.raw-links{margin-top:12px;font-size:12px}.empty{color:var(--muted);font-style:italic}
@media(max-width:700px){main{padding:0 12px 40px;margin-top:20px}.event{padding:14px}.field,.field .field{grid-template-columns:1fr;gap:3px}.stats{gap:8px}.stat{min-width:0;flex:1;padding:12px}.schema{font-size:12px}.schema th,.schema td{padding:6px}}
"""


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def label(name: str) -> str:
    return name.replace("_", " ").capitalize()


class TrajectoryRenderer:
    """A presentation-only projection: request files and event logs are never edited."""

    def __init__(self, directory: Path):
        self.directory = directory

    def read_json(self, relative: str) -> Any:
        return json.loads((self.directory / relative).read_text(encoding="utf-8"))

    def image(self, source: Any, caption: str = "Image") -> str:
        if isinstance(source, dict):
            source = source.get("asset_path", source.get("url", source.get("path", "")))
        if not isinstance(source, str):
            return self.value(source)
        if source.startswith("data:image/") and ";base64," in source:
            prefix, encoded = source.split(";base64,", 1)
            extension = prefix.split("/")[-1]
            if extension not in {"png", "jpeg", "jpg", "webp", "gif"}:
                return '<p class="empty">Image format available in the raw request.</p>'
            data = base64.b64decode(encoded, validate=True)
            source = f"images/{hashlib.sha256(data).hexdigest()}.{extension}"
            path = self.directory / source
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        if not source or source.lower().startswith(("javascript:", "data:", "vbscript:")):
            return '<p class="empty">Image reference available in the raw request.</p>'
        return (
            f'<figure><a href="{escape(source)}"><img src="{escape(source)}" '
            f'alt="{escape(caption)}" loading="lazy"></a><figcaption>{escape(caption)}</figcaption></figure>'
        )

    def text(self, text: str) -> str:
        """Keep prose verbatim; display embedded JSON as fields instead of serialized text."""
        parts: list[str] = []
        decoder = json.JSONDecoder()
        cursor = 0
        for match in re.finditer(r"[\[{]", text):
            if match.start() < cursor:
                continue
            try:
                value, length = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if match.start() > cursor:
                parts.append(f'<div class="prose">{escape(text[cursor:match.start()])}</div>')
            parts.append(self.value(value))
            cursor = match.start() + length
        if cursor < len(text):
            parts.append(f'<div class="prose">{escape(text[cursor:])}</div>')
        return "".join(parts) or '<span class="empty">Empty text</span>'

    def value(self, value: Any) -> str:
        if isinstance(value, dict):
            if value.get("type") == "function" and "name" in value and "parameters" in value:
                return self.tool(value)
            if not value:
                return '<span class="empty">Empty object</span>'
            fields = "".join(
                f'<div class="field"><dt title="{escape(key)}">{escape(label(key))}</dt>'
                f"<dd>{self.value(item)}</dd></div>"
                for key, item in value.items()
            )
            return f'<dl class="fields">{fields}</dl>'
        if isinstance(value, list):
            if not value:
                return '<span class="empty">Empty list</span>'
            if all(
                isinstance(row, list) and row and all(isinstance(v, (int, float)) for v in row)
                for row in value
            ):
                rows = "".join(
                    "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in value
                )
                return f'<div class="matrix-wrap"><table class="matrix"><tbody>{rows}</tbody></table></div>'
            if all(not isinstance(item, (dict, list)) for item in value):
                return (
                    '<div class="vector">'
                    + "".join(f"<span>{self.value(item)}</span>" for item in value)
                    + "</div>"
                )
            return '<ol class="items">' + "".join(f"<li>{self.value(item)}</li>" for item in value) + "</ol>"
        if isinstance(value, str):
            return self.text(value)
        scalar = (
            "None (null)" if value is None else str(value).lower() if isinstance(value, bool) else str(value)
        )
        return f'<span class="scalar">{escape(scalar)}</span>'

    def schema(self, schema: Object) -> str:
        if "properties" not in schema:
            return self.value(schema)
        rows = []
        for name, specification in schema["properties"].items():
            required = "Required" if name in schema.get("required", []) else "Optional"
            kind = specification.get("type", "See constraints")
            constraints = {key: value for key, value in specification.items() if key != "type"}
            rows.append(
                f"<tr><td><strong>{escape(name)}</strong></td><td>{escape(kind)}<br>"
                f'<span class="muted">{required}</span></td><td>{self.value(constraints)}</td></tr>'
            )
        remaining = {key: value for key, value in schema.items() if key not in {"properties", "required"}}
        return (
            '<table class="schema"><thead><tr><th>Field</th><th>Type</th><th>Description / constraints</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + self.value(remaining)
        )

    def tool(self, tool: Object) -> str:
        remaining = {
            key: value for key, value in tool.items() if key not in {"name", "description", "parameters"}
        }
        return (
            f'<section class="tool-card"><h4 class="tool-name">{escape(tool.get("name", "Tool"))}</h4>'
            + self.text(tool.get("description", ""))
            + self.schema(tool.get("parameters", {}))
            + self.value(remaining)
            + "</section>"
        )

    def content(self, content: Any) -> str:
        if not isinstance(content, list):
            return self.text(content) if isinstance(content, str) else self.value(content)
        parts: list[str] = []
        index = 0
        while index < len(content):
            block = content[index]
            if not isinstance(block, dict):
                parts.append(self.value(block))
                index += 1
                continue
            kind = block.get("type")
            if kind in {"input_text", "output_text", "text", "summary_text"}:
                text = block.get("text", "")
                following = content[index + 1] if index + 1 < len(content) else None
                if (
                    isinstance(following, dict)
                    and following.get("type") in {"input_image", "image_url"}
                    and len(text) <= 120
                    and "\n" not in text
                ):
                    parts.append(self.image(following.get("image_url"), text))
                    index += 2
                    continue
                parts.append(self.text(text))
            elif kind in {"input_image", "image_url"}:
                parts.append(self.image(block.get("image_url")))
            else:
                parts.append(self.value(block))
            index += 1
        return "".join(parts)

    def conversation(self, items: Any) -> str:
        if not isinstance(items, list):
            return self.content(items)
        names = {
            item.get("call_id"): item.get("name") for item in items if item.get("type") == "function_call"
        }
        parts: list[str] = []
        for index, item in enumerate(items, 1):
            kind = item.get("type", "message")
            role = item.get("role", "user")
            if kind == "function_call":
                role = "assistant"
                title = f"Assistant → {item.get('name', 'tool')}"
                body = self.content(item.get("arguments", ""))
            elif kind == "function_call_output":
                role = "tool"
                title = f"Tool result · {names.get(item.get('call_id')) or item.get('call_id', '')}"
                body = self.content(item.get("output", ""))
            elif kind == "message":
                title = label(role)
                body = self.content(item.get("content", ""))
            else:
                title, body = label(kind), self.value(item)
            identity = (
                f'<div class="meta">Call ID: {escape(item["call_id"])}</div>' if "call_id" in item else ""
            )
            parts.append(
                f'<section class="message {escape(role)}"><header class="message-header">'
                f'<span class="muted">{index:02d}</span><span>{escape(title)}</span></header>'
                f'<div class="message-body">{body}{identity}</div></section>'
            )
        return "".join(parts)

    def request(self, relative: str) -> str:
        request = self.read_json(relative)
        settings = {
            key: value
            for key, value in request.items()
            if key not in {"instructions", "input", "tools", "text"}
        }
        parts = [
            '<details class="prompt"><summary>Full prompt sent to model</summary>',
            '<details class="instructions"><summary>System instructions</summary>',
            self.text(request.get("instructions", "")),
            '</details><section class="prompt-section"><h4>Conversation · in request order</h4>',
            self.conversation(request.get("input", [])),
            "</section>",
        ]
        if request.get("tools"):
            parts.append('<details class="configuration"><summary>Available tools</summary>')
            parts.extend(self.tool(tool) for tool in request["tools"])
            parts.append("</details>")
        output_format = request.get("text", {}).get("format", {})
        if output_format:
            parts.extend(
                [
                    '<details class="configuration"><summary>Required response format</summary>',
                    self.schema(output_format.get("schema", {})),
                    self.value({key: value for key, value in output_format.items() if key != "schema"}),
                    "</details>",
                ]
            )
        parts.extend(
            [
                '<details class="configuration"><summary>Model settings</summary>',
                self.value(settings),
                "</details>",
                f'<p class="raw-links"><a href="{escape(relative)}">Exact request JSON</a></p></details>',
            ]
        )
        return "".join(parts)

    def model_call(self, event: Object, completed: Object | None) -> str:
        call = event["call_id"]
        role = label(event.get("role", "model"))
        status = (
            completed.get("provider_status", completed["kind"].removeprefix("model_"))
            if completed
            else "In progress"
        )
        parts = [
            f'<article class="event model-call" id="call-{escape(call)}"><header class="event-header">'
            f'<h3>Model call {escape(call)} · {escape(role)}</h3><span class="badge">{escape(status)}</span></header>',
            self.request(event["request_path"]),
        ]
        if completed and completed.get("response_path"):
            response = self.read_json(completed["response_path"])
            parts.extend(
                [
                    '<section class="response"><h4>Model response</h4>',
                    self.conversation(response.get("output", [])),
                    self.value({"error": response["error"]}) if "error" in response else "",
                    f'<p class="raw-links"><a href="{escape(completed["response_path"])}">Exact response JSON</a></p></section>',
                ]
            )
        if completed and completed.get("error"):
            parts.append(self.value({"error": completed["error"]}))
        parts.append("</article>")
        return "".join(parts)

    def event(self, event: Object) -> str:
        kind = event["kind"]
        if kind == "observation":
            return (
                f'<details class="event"><summary>Observation · {escape(event.get("reason", ""))}</summary>'
                '<div class="images">'
                + "".join(self.image(image["path"], image["label"]) for image in event.get("images", []))
                + "</div></details>"
            )
        if kind == "action_completed":
            action = event["action"]
            return (
                f'<article class="event action"><h3>Executed · {escape(action["action"])}</h3>'
                + self.text(action["rationale"])
                + f'<div class="meta">Source: {escape(action["source"])} · {escape(action["control_steps"])} control steps</div></article>'
            )
        if kind in {
            "control_step",
            "action_started",
            "action_proposed",
            "model_completed",
            "model_failed",
            "planning_agent_started",
            "planning_agent_completed",
            "episode_finished",
            "plugin_started",
        }:
            return ""  # These remain in the log; model cards and executed actions show their outcome.
        title = f'{label(kind)} · {event.get("plugin", "")}'.rstrip(" ·")
        values = {
            key: value
            for key, value in event.items()
            if key not in {"kind", "time", "schema_version", "sequence", "decision_id", "context_blocks"}
        }
        blocks = "".join(
            self.image(block["image"]["path"], block["image"]["label"])
            if "image" in block
            else self.text(block.get("text", ""))
            for block in event.get("context_blocks", [])
        )
        return (
            f'<details class="event"><summary>{escape(title)}</summary>{self.value(values)}{blocks}</details>'
        )

    def render(self, events: list[Object], summary: Object, metadata: Object) -> str:
        by_decision: dict[int, list[Object]] = defaultdict(list)
        completed = {
            event["call_id"]: event
            for event in events
            if event["kind"] in {"model_completed", "model_failed"}
        }
        for event in events:
            by_decision[event.get("decision_id", 0)].append(event)
        task = metadata.get("task", {})
        calls = sum(event["kind"] == "model_started" for event in events)
        outcome = (
            "Succeeded"
            if summary.get("success")
            else "Running"
            if summary.get("status") == "running"
            else "Not completed"
        )
        parts = [
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Robot trajectory · {escape(task.get("suite", "episode"))}</title><style>{STYLE}</style></head><body><main>',
            '<header><div class="eyebrow">Robot execution history</div><h1>Episode trajectory</h1>',
            f'<p>{escape(task.get("instruction", ""))}</p>',
            f'<div class="meta">{escape(task.get("suite", ""))} · Task {escape(task.get("task_id", "—"))} · {escape(self.directory.name)}</div>',
            '<div class="stats">',
            f'<div class="stat"><span class="muted">Outcome</span><strong>{outcome}</strong></div>',
            f'<div class="stat"><span class="muted">Decisions</span><strong>{escape(summary.get("decisions", "—"))}</strong></div>',
            f'<div class="stat"><span class="muted">Model calls</span><strong>{calls}</strong></div></div>',
            '<div class="links"><a href="events.jsonl">Complete event log</a><a href="summary.json">Run summary</a>',
            '<a href="metadata.json">Configuration</a></div></header><nav aria-label="Decisions">',
        ]
        for decision in sorted(by_decision):
            title = f"Decision {decision}" if decision else "Initial state"
            parts.append(f'<a href="#decision-{decision}">{title}</a>')
        parts.append("</nav>")
        for decision, group in sorted(by_decision.items()):
            title = f"Decision {decision}" if decision else "Initial state"
            parts.append(f'<section class="decision" id="decision-{decision}"><h2>{title}</h2>')
            for event in group:
                if event["kind"] == "model_started":
                    parts.append(self.model_call(event, completed.get(event["call_id"])))
                else:
                    parts.append(self.event(event))
            parts.append("</section>")
        parts.append("</main></body></html>")
        return "\n".join(parts)


def render_trajectory(source: Path) -> Path:
    """Convert an episode directory or its events.jsonl to trajectory.html."""
    directory = source.parent if source.is_file() else source
    events: list[Object] = []
    log = directory / "events.jsonl"
    if log.exists():
        lines = log.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                # An in-progress writer may not have completed the last event yet.
    renderer = TrajectoryRenderer(directory)
    summary = renderer.read_json("summary.json") if (directory / "summary.json").exists() else {}
    metadata = renderer.read_json("metadata.json") if (directory / "metadata.json").exists() else {}
    destination = directory / "trajectory.html"
    destination.write_text(renderer.render(events, summary, metadata), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert an episode log into a readable offline HTML page.")
    parser.add_argument("episode", type=Path, help="Episode directory or its events.jsonl file")
    args = parser.parse_args()
    source = args.episode.expanduser().resolve()
    directory = source.parent if source.is_file() else source
    if not (directory / "events.jsonl").is_file():
        parser.error(f"No events.jsonl found in {directory}")
    print(render_trajectory(source))
    return 0
