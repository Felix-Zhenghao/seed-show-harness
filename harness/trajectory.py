"""Incremental, inspectable episode records with content-addressed image assets."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness.domain import ImageAsset, Observation
from harness.serialization import Object, dumps, jsonable
from harness.trajectory_viewer import render_trajectory


class TrajectoryStore:
    def __init__(self, output_root: Path, suite: str, task_id: int, secrets: tuple[str, ...] = ()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = output_root / suite / f"task_{task_id:03d}" / f"{stamp}_{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        for name in ("images", "calls", "helpers"):
            (self.directory / name).mkdir()
        self.history_path = self.directory / "events.jsonl"
        self._events = self.history_path.open("a", encoding="utf-8")
        self._lock = threading.RLock()
        self._sequence = 0
        self._secrets = tuple(secret for secret in secrets if secret)
        self._closed = False
        self.write_json("summary.json", {"status": "running"})
        render_trajectory(self.directory)

    def _scrub(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "[REDACTED]"
                    if key.lower()
                    in {"api_key", "authorization", "x-api-key", "access_token", "password", "secret"}
                    else self._scrub(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._scrub(item) for item in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    def write_json(self, relative: str, value: Any) -> str:
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._lock:
            temporary.write_text(dumps(self._scrub(self._with_assets(value))) + "\n", encoding="utf-8")
            temporary.replace(path)
        return relative

    def record(self, kind: str, **data: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Trajectory is closed")
            self._sequence += 1
            event = {
                "schema_version": 1,
                "sequence": self._sequence,
                "time": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                **data,
            }
            self._events.write(dumps(self._scrub(self._with_assets(event))) + "\n")
            self._events.flush()

    def _with_assets(self, value: Any) -> Any:
        if isinstance(value, ImageAsset):
            return {
                "label": value.label,
                "path": self.save_image(value),
                "width": value.width,
                "height": value.height,
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: self._with_assets(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        if isinstance(value, Mapping):
            return {key: self._with_assets(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._with_assets(item) for item in value]
        return jsonable(value)

    def save_image(self, image: ImageAsset) -> str:
        relative = f"images/{image.digest}.png"
        path = self.directory / relative
        if not path.exists():
            path.write_bytes(image.png)
        return relative

    def observation(self, observation: Observation, decision_id: int, reason: str) -> None:
        images = [
            {
                "label": image.label,
                "path": self.save_image(image),
                "width": image.width,
                "height": image.height,
            }
            for image in observation.images
        ]
        self.record(
            "observation",
            decision_id=decision_id,
            observation_id=observation.sequence,
            reason=reason,
            images=images,
            cameras=observation.cameras,
            robot=observation.robot,
            success=observation.success,
            simulation_time=observation.simulation_time,
            simulator_state=observation.simulator_state,
        )

    def externalize(self, value: Any) -> Any:
        """Replace image data URLs with exact-byte local references in wire logs."""
        if isinstance(value, dict):
            return {key: self.externalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.externalize(item) for item in value]
        if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
            prefix, data = value.split(";base64,", 1)
            raw = base64.b64decode(data, validate=True)
            extension = prefix.split("/")[-1]
            relative = f"images/{hashlib.sha256(raw).hexdigest()}.{extension}"
            path = self.directory / relative
            if not path.exists():
                path.write_bytes(raw)
            return {"asset_path": relative, "data_url_prefix": prefix + ";base64,"}
        return value

    def finish(self, summary: Object) -> None:
        self.record("episode_finished", **summary)
        self.write_json("summary.json", summary)
        render_trajectory(self.directory)
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._events.flush()
                os.fsync(self._events.fileno())
                self._events.close()
                self._closed = True
