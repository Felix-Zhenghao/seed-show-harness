"""Validated run settings, independent of live episode state."""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml

ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class EnvironmentConfig:
    name: str = "libero"
    libero_root: Path = ROOT.parent / "third_party" / "LIBERO"
    suite: str = "libero_10"
    task_id: int = 0
    task_order_index: int = 0
    initial_state: int = 0
    seed: int = 0
    image_size: int = 256
    control_hz: int = 20
    settle_steps: int = 10
    image_rotation: int = 180
    gl_backend: str = "auto"


@dataclasses.dataclass(frozen=True)
class ActionConfig:
    translation_m: float = 0.02
    rotation_deg: float = 15.0
    motion_steps: int = 10
    gripper_steps: int = 15
    empty_gripper_width_m: float = 0.005


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com/beta"
    secrets_file: Path = ROOT.parent / "configs" / "secrets.local.env"
    api_key_env: str = "DEEPSEEK_API_KEY"
    reasoning_effort: str = "none"
    max_output_tokens: int = 16384
    timeout_s: float = 180.0
    transport_retries: int = 2
    dsh_executable: str = "dsh"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    environment: EnvironmentConfig = dataclasses.field(default_factory=EnvironmentConfig)
    actions: ActionConfig = dataclasses.field(default_factory=ActionConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    output_root: Path = ROOT / "runs"
    max_decisions: int = 20
    max_tool_calls_per_decision: int | None = None
    max_output_corrections: int = 2
    history_length: int = 10
    plugins: tuple[str, ...] = ("base_prompt", "action_mem", "subgoal", "affordance", "auto_release")

    def validate(self) -> None:
        for value, name, minimum in [
            (self.max_decisions, "max_decisions", 1),
            (self.history_length, "history_length", 1),
            (self.max_output_corrections, "max_output_corrections", 0),
            (self.environment.task_id, "task_id", 0),
            (self.environment.task_order_index, "task_order_index", 0),
            (self.environment.initial_state, "initial_state", 0),
            (self.environment.seed, "seed", 0),
            (self.environment.image_size, "image_size", 1),
            (self.environment.control_hz, "control_hz", 1),
            (self.environment.settle_steps, "settle_steps", 0),
            (self.actions.motion_steps, "motion_steps", 1),
            (self.actions.gripper_steps, "gripper_steps", 1),
            (self.model.max_output_tokens, "max_output_tokens", 1),
            (self.model.transport_retries, "transport_retries", 0),
        ]:
            if (not isinstance(value, int) or isinstance(value, bool)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        budget = self.max_tool_calls_per_decision
        if budget is not None and ((not isinstance(budget, int) or isinstance(budget, bool)) or budget < 0):
            raise ValueError("Tool budget must be null or a nonnegative integer")
        for number, name, maximum in [
            (self.actions.translation_m, "translation_m", 1.0),
            (self.actions.rotation_deg, "rotation_deg", 180.0),
            (self.actions.empty_gripper_width_m, "empty_gripper_width_m", 0.1),
            (self.model.timeout_s, "timeout_s", 3600.0),
        ]:
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not 0 < number <= maximum
            ):
                raise ValueError(f"{name} must be finite in (0, {maximum}]")
        if self.environment.name == "libero" and self.environment.image_rotation != 180:
            raise ValueError("The LIBERO action vocabulary requires 180-degree image rotation")
        if self.model.model != "deepseek-flash":
            raise ValueError("This runtime requires deepseek-flash")
        if self.model.reasoning_effort not in ("none", "low", "high", "max"):
            raise ValueError("Invalid reasoning_effort")
        if len(set(self.plugins)) != len(self.plugins):
            raise ValueError("Duplicate enabled plugin")


def _section(cls: type[T], values: dict[str, Any], directory: Path) -> T:
    fields = {field.name: field for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
    if unknown := set(values) - set(fields):
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    result = dict(values)
    for name in ("libero_root", "secrets_file", "output_root"):
        if name in result:
            path = Path(result[name]).expanduser()
            result[name] = (directory / path).resolve() if not path.is_absolute() else path
    if "plugins" in result:
        result["plugins"] = tuple(result["plugins"])
    return cls(**result)


def load_config(path: Path | None = None) -> RunConfig:
    data = yaml.safe_load(path.read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    directory = path.resolve().parent if path else ROOT
    sections: dict[str, Any] = {}
    for name, cls in (("environment", EnvironmentConfig), ("actions", ActionConfig), ("model", ModelConfig)):
        sections[name] = _section(cls, data.pop(name, {}), directory)
    config = _section(RunConfig, {**data, **sections}, directory)
    config.validate()
    return config


def load_api_key(config: ModelConfig) -> str:
    if value := os.environ.get(config.api_key_env):
        return value
    if config.secrets_file.is_file():
        for line in config.secrets_file.read_text().splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == config.api_key_env:
                value = value.strip().strip("\"'")
                if value:
                    return value
    raise ValueError(f"Missing credential {config.api_key_env}")
