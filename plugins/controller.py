"""Dependency scheduling, result lifetimes, and shared invocation accounting."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal, Mapping

from harness.content import ImageBlock
from harness.context import ContextUpdater, PluginContext
from harness.ports import EventSink
from harness.schemas import check_schema, validate
from plugins.contracts import PluginResult, PluginSpec, StoredResult


def validate_registry(registry: Mapping[str, PluginSpec]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name not in registry:
            raise ValueError(f"Unknown or disabled dependency: {name}")
        if name in visiting:
            raise ValueError(f"Dependency cycle at {name}")
        if name in visited:
            return
        plugin = registry[name]
        if plugin.name != name:
            raise ValueError(f"Registry key does not match {name}")
        if plugin.activation not in ("model_call", "every_decision", "conditional"):
            raise ValueError(f"Invalid activation: {name}")
        if plugin.lifetime not in ("decision", "until_next_call"):
            raise ValueError(f"Invalid lifetime: {name}")
        if (plugin.activation == "conditional") != (plugin.condition is not None):
            raise ValueError(f"Only conditional plugins require a condition: {name}")
        if len({edge.tool for edge in plugin.dependencies}) != len(plugin.dependencies):
            raise ValueError(f"Duplicate dependency: {name}")
        check_schema(plugin.parameters)
        check_schema(plugin.result_schema)
        visiting.add(name)
        for edge in plugin.dependencies:
            visit(edge.tool)
        visiting.remove(name)
        visited.add(name)

    for name in registry:
        visit(name)


class PluginController:
    def __init__(self, registry: Mapping[str, PluginSpec], updater: ContextUpdater, events: EventSink):
        validate_registry(registry)
        self.registry = MappingProxyType(dict(registry))
        self.updater, self.events = updater, events
        self._versions: dict[str, int] = {}
        self._dependents: dict[str, list[str]] = {name: [] for name in registry}
        self._invocation_sequence = 0
        for plugin in registry.values():
            for dependency in plugin.dependencies:
                self._dependents[dependency.tool].append(plugin.name)

    def reusable(self, name: str) -> bool:
        state = self.updater.state
        record = state.results.get(name)
        if record is None:
            return False
        if self.registry[name].lifetime == "until_next_call":
            return True
        return (
            record.decision_id == state.decision_id
            and record.observation_revision == state.observation_revision
            and all(
                self.reusable(dep) and state.results[dep].version == version
                for dep, version in record.dependency_versions.items()
            )
        )

    def retire(self, name: str) -> StoredResult | None:
        previous = self.updater.state.results.get(name)
        pending, seen = [name], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in self.updater.state.results:
                self.updater.set_result(current, None)
            pending.extend(
                child for child in self._dependents[current] if self.registry[child].lifetime == "decision"
            )
        return previous

    def expire_decision_results(self) -> None:
        for name in tuple(self.updater.state.results):
            if name in self.updater.state.results and self.registry[name].lifetime == "decision":
                self.retire(name)

    def _save(self, plugin: PluginSpec, result: PluginResult, versions: Mapping[str, int]) -> None:
        if not isinstance(result, PluginResult):
            raise TypeError(f"{plugin.name} did not return PluginResult")
        validate(plugin.result_schema, result.data)
        plugin.validate_result(result.data)
        replaced_views: set[str] = set()
        for block in result.context_blocks:
            if isinstance(block, ImageBlock) and block.replaces_observation is not None:
                self.updater.state.observation.image(block.replaces_observation)
                if block.replaces_observation in replaced_views:
                    raise ValueError(f"Multiple replacements for observation {block.replaces_observation!r}")
                replaced_views.add(block.replaces_observation)
        if result.observations_changed:
            if plugin.activation != "conditional":
                raise ValueError("Only conditional interventions may change observations")
            self.expire_decision_results()
        if result.handled_event_id is not None:
            if plugin.activation != "conditional":
                raise ValueError("Only interventions may handle events")
            self.updater.mark_handled(plugin.name, result.handled_event_id)
        version = self._versions.get(plugin.name, 0) + 1
        self._versions[plugin.name] = version
        state = self.updater.state
        self.updater.set_result(
            plugin.name,
            StoredResult(
                result,
                version,
                state.decision_id,
                state.observation_revision,
                MappingProxyType(dict(versions)),
            ),
        )

    def invoke(
        self,
        name: str,
        arguments: Any,
        origin: Literal["model", "dependency", "automatic"] = "model",
        parent_invocation: int | None = None,
    ) -> PluginResult:
        self._invocation_sequence += 1
        invocation = self._invocation_sequence
        details = {
            "plugin": name,
            "arguments": arguments,
            "origin": origin,
            "invocation_id": invocation,
            "parent_invocation": parent_invocation,
            "decision_id": self.updater.state.decision_id,
        }
        plugin = self.registry.get(name)
        record_invocation = plugin is None or plugin.activation == "model_call"
        if record_invocation:
            self.events.record("plugin_started", **details)
        try:
            if origin != "automatic":
                self.updater.reserve_call()
            plugin = self.registry[name]
            if origin == "model" and plugin.activation != "model_call":
                raise ValueError(f"Not an information tool: {name}")
            if plugin.condition is not None and not plugin.condition(PluginContext(self.updater.state)):
                raise RuntimeError(f"Required condition is false: {name}")
            validate(plugin.parameters, arguments)
            plugin.validate_arguments(arguments)
            previous = self.retire(name)
            inputs: dict[str, PluginResult] = {}
            versions: dict[str, int] = {}
            for edge in plugin.dependencies:
                if not self.reusable(edge.tool):
                    context = PluginContext(self.updater.state, previous.result.data if previous else None)
                    self.invoke(
                        edge.tool,
                        edge.build_arguments(arguments, context),
                        "dependency",
                        invocation if record_invocation else None,
                    )
                record = self.updater.state.results[edge.tool]
                inputs[edge.tool], versions[edge.tool] = record.result, record.version
            revision = self.updater.state.observation_revision
            result = plugin.handler(
                arguments,
                PluginContext(self.updater.state, previous.result.data if previous else None),
                MappingProxyType(inputs),
            )
            if not isinstance(result, PluginResult):
                raise TypeError(f"{plugin.name} did not return PluginResult")
            if result.observations_changed and self.updater.state.observation_revision == revision:
                raise RuntimeError("Intervention must install fresh observations before returning")
            self._save(plugin, result, versions)
            if record_invocation:
                self.events.record(
                    "plugin_completed", **details, result=result.data, context_blocks=result.context_blocks
                )
            return result
        except Exception as error:
            if record_invocation:
                self.events.record("plugin_failed", **details, error=str(error))
            raise

    def begin_decision(self, budget: int | None = None) -> None:
        self.updater.begin_decision(budget)
        self.expire_decision_results()
        for activation in ("conditional", "every_decision"):
            for plugin in self.registry.values():
                if plugin.activation != activation:
                    continue
                if plugin.condition is not None and not plugin.condition(PluginContext(self.updater.state)):
                    continue
                record = self.updater.state.results.get(plugin.name)
                if (
                    record is not None
                    and record.decision_id == self.updater.state.decision_id
                    and self.reusable(plugin.name)
                ):
                    continue  # A shared prerequisite already ran this automatic plugin.
                self.invoke(plugin.name, {}, origin="automatic")
