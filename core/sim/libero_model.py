"""Hosted-model request dialects and an audit trail of actual model inputs/outputs."""
from __future__ import annotations

import base64
import copy
import html
import json
import os
import time
from pathlib import Path

from core.secrets import load_secrets_env
from core.record.episode_logger import _jsonable, _redact_sensitive_metadata
from core.vlm.vlm_client import VLMClient


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(_redact_sensitive_metadata(value)), indent=2, ensure_ascii=False))


class DebugTrace:
    HEADER = (
        '<!doctype html><meta charset="utf-8"><title>LIBERO trajectory</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:16px system-ui;padding:0 20px;color:#223044}'
        'pre{white-space:pre-wrap;background:#f4f5f7;padding:16px;overflow-wrap:anywhere}'
        'img{max-width:43%;margin:8px;border-radius:8px}article{border-top:1px solid #ccc;padding:16px 0}'
        'summary{cursor:pointer;color:#3157a0;margin:10px 0}h2{font-size:21px}.answer{background:#eef6f3;padding:16px}'
        'table{border-collapse:collapse}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>'
        '<h1>LIBERO harness trajectory</h1><p>Read each observation, model decision, and executed action in order. '
        'Expand a record for exact prompts, returned reasoning, control commands, and measured state.</p>'
    )

    def __init__(self, directory: Path, enabled: bool):
        self.directory, self.enabled = directory, enabled
        self.calls = 0
        self.context = {}
        if enabled:
            (directory / "calls").mkdir()
            (directory / "trajectory.html").write_text(self.HEADER)

    def event(self, kind: str, data: dict, images=()) -> None:
        if not self.enabled:
            return
        item = _jsonable(_redact_sensitive_metadata({"event": kind, **data, "images": list(images)}))
        with (self.directory / "trace.jsonl").open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (self.directory / "trajectory.html").open("a") as f:
            f.write(self.render_event(item))

    @staticmethod
    def render_event(item: dict) -> str:
        escape = lambda value: html.escape(str(value))
        kind = item["event"]
        label = kind
        if "role" in item:
            label = f"{item['role'].title()} · step {item.get('step', 0)} · call {item.get('call', '?')} · {kind}"
        parts = [f"<article><h2>{escape(label)}</h2>"]
        for path in item.get("images", []):
            parts.append(f'<a href="{escape(path)}"><img src="{escape(path)}" alt="observation"></a>')
        if kind == "Model input":
            prompts = []
            for message in item["request"]["messages"]:
                content = message.get("content", "")
                if isinstance(content, str):
                    prompts.append(content)
                else:
                    prompts.extend(block["text"] for block in content if block.get("type") == "text")
            parts.append("<details><summary>Exact model prompt</summary><pre>" +
                         escape("\n\n".join(prompts)) + "</pre></details>")
        if kind == "Model output":
            response = item["response"]
            message = response["choices"][0]["message"]
            parts.append(f"<p>{item['latency_s']:.2f} seconds · "
                         f"{response.get('usage', {}).get('total_tokens', '?')} total tokens</p>")
            parts.append('<pre class="answer">' + escape(message.get("content", "")) + "</pre>")
            if message.get("reasoning_content"):
                parts.append("<details><summary>Provider reasoning</summary><pre>" +
                             escape(message["reasoning_content"]) + "</pre></details>")
        if "action_source" in item:
            if item.get("auto_release"):
                parts.append('<p class="answer">Automatic RELEASE: the measured closed gripper was empty and was reopened.</p>')
            parts.append(f"<p>Source: <b>{escape(item['action_source'])}</b> · "
                         f"Stage: {escape(item['stage']['description'])} · "
                         f"Simulator success: {escape(item['success'])}</p>")
            parts.append('<p class="answer">' + escape(item["vlm"]["c"]["why"]) + "</p>")
            parts.append("<table><tr><th>Measurement</th><th>Before</th><th>After</th></tr>")
            for key in ("eef_pos", "gripper_width_m", "camera_world_position"):
                parts.append(f"<tr><td>{escape(key)}</td><td>{escape(item['before'][key])}</td>"
                             f"<td>{escape(item['after'][key])}</td></tr>")
            parts.append("</table>")
        if kind in ("Summary", "Episode stopped", "Model error"):
            parts.append('<pre class="answer">' + escape(json.dumps(item, indent=2)) + "</pre>")
        parts.append("<details><summary>Full structured record</summary><pre>" +
                     escape(json.dumps(item, indent=2, ensure_ascii=False)) + "</pre></details></article>")
        return "".join(parts)

    def request(self, payload: dict) -> tuple[int, list]:
        self.calls += 1
        call = self.calls
        images = []
        if self.enabled:
            request = copy.deepcopy(payload)
            for message in request["messages"]:
                if isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if block.get("type") == "image_url":
                            url = block["image_url"]["url"]
                            if url.startswith("data:image/"):
                                relative = f"calls/{call:04d}_image_{len(images)}.png"
                                (self.directory / relative).write_bytes(base64.b64decode(url.split(",", 1)[1]))
                                block["image_url"]["url"] = relative
                                images.append(relative)
            write_json(self.directory / "calls" / f"{call:04d}_request.json", request)
            self.event("Model input", {"call": call, **self.context, "request": request}, images)
        return call, images


class LiberoVLMClient(VLMClient):
    def __init__(self, cfg: dict, trace: DebugTrace):
        model = cfg["model"]
        kimi = model["name"] == "kimi-k3"
        load_secrets_env()  # Optional existing, gitignored credential store; no config overlays.
        env_name = model["api_key_env"] or ("MOONSHOT_API_KEY" if kimi else "DEEPSEEK_API_KEY")
        key = model["api_key"] or os.environ.get(env_name)
        if not key:
            raise ValueError(f"Set model.api_key in context.py or export {env_name}")
        self.settings, self.trace = model, trace
        super().__init__(
            base_url=model["base_url"] or ("https://api.moonshot.ai/v1" if kimi else "https://api.deepseek.com"),
            model=model["name"], api_key=key, timeout_s=model["timeout_s"], max_tokens=model["max_tokens"],
            temperature=model["temperature"], max_retries=model["max_retries"],
            retry_base_delay_s=model["retry_base_delay_s"], retry_max_delay_s=model["retry_max_delay_s"],
        )

    def _finalize_payload(self, payload: dict) -> dict:
        kimi = self.model == "kimi-k3"
        out = {"model": self.model, "messages": payload["messages"],
               "max_completion_tokens" if kimi else "max_tokens": payload.get("max_tokens", self.max_tokens)}
        if kimi:
            out["reasoning_effort"] = self.settings["reasoning_effort"]
        else:
            out["thinking"] = {"type": "enabled" if self.settings["thinking"] else "disabled"}
            if self.settings["thinking"]:
                out["reasoning_effort"] = self.settings["reasoning_effort"]
            else:
                out["temperature"] = self.temperature
        # Keep Kimi prompt-guided so planner repair calls and token retries share
        # the same protocol. The harness validates the returned JSON and vocabulary.
        if "response_format" in payload and not kimi:
            out["response_format"] = {"type": "json_object"}
        return out

    def _post_chat(self, payload: dict) -> dict:
        call, _ = self.trace.request(payload)
        started = time.monotonic()
        try:
            result = super()._post_chat(payload)
        except BaseException as exc:
            error = str(exc).replace(self._api_key, "[REDACTED]")
            self.trace.event("Model error", {"call": call, **self.trace.context, "error": error})
            # Do not allow the generic controller's parse fallback to turn a broken
            # API/authentication into an unrequested robot motion.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ConnectionError(error) from None
        if self.trace.enabled:
            write_json(self.trace.directory / "calls" / f"{call:04d}_response.json", result)
            self.trace.event("Model output", {"call": call, **self.trace.context,
                                              "latency_s": time.monotonic() - started, "response": result})
        if not result["choices"][0]["message"].get("content"):
            raise ConnectionError("Model returned no final answer; increase model.max_tokens or "
                                  "harness.planner_max_tokens. Provider reasoning is not an action.")
        return result
