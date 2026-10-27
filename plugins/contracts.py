"""Plugin-owned schemas and behavior; no simulator or provider dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

from harness.content import ContentBlock
from harness.context import PluginContext
from harness.serialization import Object, freeze

Validator = Callable[[Mapping[str, Any]], None]


def no_validation(data: Mapping[str, Any]) -> None:
    pass


@dataclass(frozen=True)
class Dependency:
    tool: str
    build_arguments: Callable[[Mapping[str, Any], PluginContext], Object]


@dataclass(frozen=True)
class PluginResult:
    """Keep structured data for dependencies/logs and author model-facing guidance in context_blocks."""

    data: Mapping[str, Any]
    context_blocks: tuple[ContentBlock, ...] = ()
    observations_changed: bool = False
    handled_event_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", freeze(self.data))


@dataclass(frozen=True)
class PluginSpec:
    name: str
    activation: Literal["every_decision", "model_call", "conditional"]
    lifetime: Literal["decision", "until_next_call"]
    description: str
    parameters: Object
    result_schema: Object
    handler: Callable[[Mapping[str, Any], PluginContext, Mapping[str, PluginResult]], PluginResult]
    dependencies: tuple[Dependency, ...] = ()
    condition: Callable[[PluginContext], bool] | None = None
    validate_arguments: Validator = no_validation
    validate_result: Validator = no_validation
    presentation: Literal["instructions", "context"] = "context"


@dataclass(frozen=True)
class StoredResult:
    result: PluginResult
    version: int
    decision_id: int
    observation_revision: int
    dependency_versions: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))


def tool_definitions(registry: Mapping[str, PluginSpec]) -> list[Object]:
    """Register callable plugins through the native DeepSeek Responses tools parameter."""
    return [
        {
            "type": "function",
            "name": plugin.name,
            "description": plugin.description,
            "parameters": plugin.parameters,
            "strict": True,
        }
        for _, plugin in sorted(registry.items())
        if plugin.activation == "model_call"
    ]
