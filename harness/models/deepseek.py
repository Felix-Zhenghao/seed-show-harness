"""One strict Responses transport shared by direct VLMs and the planning agent."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import requests

from harness.config import ModelConfig
from harness.models.streaming import response_events
from harness.schemas import check_schema
from harness.serialization import Object, dumps
from harness.trajectory import TrajectoryStore


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


class IncompleteModelResponse(ValueError):
    """A truncated response is eligible for output correction, never execution."""


@dataclass(frozen=True)
class ModelReply:
    text: str
    tool_calls: tuple[ToolCall, ...]
    output_items: tuple[Object, ...]
    raw: Object

    @classmethod
    def from_response(cls, data: Object) -> ModelReply:
        items = data.get("output", [])
        if not isinstance(items, list):
            raise ValueError("Provider output must be an array")
        calls: list[ToolCall] = []
        call_ids: set[str] = set()
        text: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Provider output items must be objects")
            if item.get("type") == "function_call":
                call_id, name, arguments = (item.get(key) for key in ("call_id", "name", "arguments"))
                if not isinstance(call_id, str) or not call_id.strip() or call_id in call_ids:
                    raise ValueError("Native tool calls require nonempty, unique call IDs")
                if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
                    raise ValueError("Native tool calls require a name and JSON-encoded arguments")
                calls.append(ToolCall(call_id, name, arguments))
                call_ids.add(call_id)
            elif item.get("type") == "message":
                text.extend(
                    block["text"] for block in item.get("content", []) if block.get("type") == "output_text"
                )
        return cls("".join(text), tuple(calls), tuple(items), data)

    def with_tool_results(self, results: Mapping[str, str | list[Object]]) -> list[Object]:
        """Replay the actual model output and pair each native call with its result."""
        if set(results) != {call.call_id for call in self.tool_calls}:
            raise ValueError("Tool results must match the model's native call IDs exactly")
        return [
            *self.output_items,
            *(
                {"type": "function_call_output", "call_id": call.call_id, "output": results[call.call_id]}
                for call in self.tool_calls
            ),
        ]


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        role: str,
        instructions: str,
        input_items: Sequence[Object],
        output_schema: Object,
        tools: Sequence[Object] = (),
        tool_choice: str = "auto",
        decision_id: int = 0,
    ) -> ModelReply:
        ...


class DeepSeekClient:
    def __init__(self, config: ModelConfig, api_key: str, trajectory: TrajectoryStore):
        self.config, self.trajectory = config, trajectory
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        self._sequence = 0
        self._lock = threading.Lock()

    @property
    def model_call_count(self) -> int:
        return self._sequence

    def complete(
        self,
        *,
        role: str,
        instructions: str,
        input_items: Sequence[Object],
        output_schema: Object,
        tools: Sequence[Object] = (),
        tool_choice: str = "auto",
        decision_id: int = 0,
    ) -> ModelReply:
        check_schema(output_schema)
        definitions = sorted(tools, key=lambda tool: tool["name"])
        for tool in definitions:
            if tool.get("strict") is not True:
                raise ValueError("Every model tool must enable strict mode")
            check_schema(tool["parameters"])
        payload: Object = {
            "model": self.config.model,
            "instructions": instructions,
            "input": list(input_items),
            "reasoning": {"effort": self.config.reasoning_effort},
            "max_output_tokens": self.config.max_output_tokens,
            "stream": True,
            "text": {
                "format": {"type": "json_schema", "name": role, "strict": True, "schema": output_schema}
            },
            "tools": definitions,
            "tool_choice": tool_choice if definitions else "none",
        }
        if self.config.reasoning_effort == "none":
            payload["temperature"] = 0
        return self._request(role, payload, decision_id)

    def _request(self, role: str, payload: Object, decision_id: int) -> ModelReply:
        for attempt in range(self.config.transport_retries + 1):
            with self._lock:
                self._sequence += 1
                call_id = self._sequence
            request_path = self.trajectory.write_json(f"calls/{call_id:05d}_{role}_request.json", payload)
            stream_path = f"calls/{call_id:05d}_{role}_stream.jsonl"
            self.trajectory.record(
                "model_started",
                role=role,
                call_id=call_id,
                decision_id=decision_id,
                attempt=attempt,
                request_path=request_path,
                stream_path=stream_path,
            )
            retry_after = 0.0
            try:
                response = self._session.post(
                    self.config.base_url.rstrip("/") + "/responses",
                    json=payload,
                    stream=True,
                    timeout=(20, self.config.timeout_s),
                )
                try:
                    data = self._read_response(response, stream_path)
                finally:
                    response.close()
                response_path = self.trajectory.write_json(f"calls/{call_id:05d}_{role}_response.json", data)
                self.trajectory.record(
                    "model_completed",
                    role=role,
                    call_id=call_id,
                    decision_id=decision_id,
                    request_path=request_path,
                    response_path=response_path,
                    http_status=response.status_code,
                    provider_status=data.get("status"),
                )
                if response.status_code >= 400:
                    error_message = dumps(data).replace(self._api_key, "[REDACTED]")
                    if response.status_code not in (408, 409, 425, 429) and response.status_code < 500:
                        raise ValueError(f"DeepSeek HTTP {response.status_code}: {error_message}")
                    try:
                        retry_after = min(float(response.headers.get("Retry-After", "0")), 90.0)
                    except ValueError:
                        retry_after = 0.0
                    raise RuntimeError(f"Transient provider HTTP {response.status_code}: {error_message}")
                if data.get("status") == "incomplete":
                    raise IncompleteModelResponse(
                        f"Response was truncated: {data.get('incomplete_details')}. "
                        "Provide one concise, complete tool call or final action JSON."
                    )
                if data.get("status") != "completed":
                    raise ValueError(
                        f"Provider did not complete: {data.get('status')}: {data.get('incomplete_details')}"
                    )
                return ModelReply.from_response(data)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                self.trajectory.record(
                    "model_failed",
                    role=role,
                    call_id=call_id,
                    decision_id=decision_id,
                    error=str(error),
                )
                if isinstance(error, ValueError) or attempt == self.config.transport_retries:
                    raise
                time.sleep(max(retry_after, min(2**attempt, 8)))
        raise AssertionError("Unreachable request loop")

    def _read_response(self, response: requests.Response, stream_path: str) -> Object:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            try:
                data = response.json()
            except ValueError as error:
                raise RuntimeError(f"Provider returned non-JSON HTTP {response.status_code}") from error
            if not isinstance(data, dict):
                raise ValueError("Provider response must be an object")
            return data
        response.encoding = "utf-8"
        with (self.trajectory.directory / stream_path).open("w", encoding="utf-8") as stream_log:
            for event in response_events(response.iter_lines(decode_unicode=True)):
                stream_log.write(dumps(event).replace(self._api_key, "[REDACTED]") + "\n")
                stream_log.flush()
                kind = event.get("type", event.get("event"))
                if kind in ("response.completed", "response.incomplete", "response.failed"):
                    result = event.get("response")
                    if not isinstance(result, dict):
                        raise ValueError("Terminal stream event has no response object")
                    return result
                if kind == "error":
                    raise RuntimeError(f"Provider stream error: {dumps(event)}")
        raise RuntimeError("Provider stream ended without a terminal response")

    def close(self) -> None:
        self._session.close()
