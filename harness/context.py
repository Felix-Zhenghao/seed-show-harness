"""Immediate episode state and the single owner of its updates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from harness.domain import ExecutedAction, GraspEvent, Observation, TaskMetadata

if TYPE_CHECKING:
    from plugins.contracts import StoredResult


@dataclass(frozen=True)
class EpisodeContext:
    task: TaskMetadata
    observation: Observation
    history_path: Path
    decision_id: int = 0
    observation_revision: int = 1
    calls_left: int | None = None
    recent_actions: tuple[ExecutedAction, ...] = ()
    latest_settled_grasp: GraspEvent | None = None
    handled_events: frozenset[tuple[str, str]] = frozenset()
    results: Mapping[str, StoredResult] = field(default_factory=lambda: MappingProxyType({}))
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginContext:
    state: EpisodeContext
    previous_result: Mapping[str, Any] | None = None


class ContextUpdater:
    def __init__(self, initial: EpisodeContext, history_length: int = 10):
        self._state = initial
        self._history_length = history_length

    @property
    def state(self) -> EpisodeContext:
        return self._state

    def begin_decision(self, budget: int | None) -> None:
        if budget is not None and ((not isinstance(budget, int) or isinstance(budget, bool)) or budget < 0):
            raise ValueError("Tool budget must be null or a nonnegative integer")
        self._state = replace(
            self._state, decision_id=self._state.decision_id + 1, calls_left=budget, errors=()
        )

    def reserve_call(self) -> None:
        remaining = self._state.calls_left
        if remaining is not None:
            if remaining == 0:
                raise RuntimeError("Tool-call budget exhausted")
            self._state = replace(self._state, calls_left=remaining - 1)

    def observe(self, observation: Observation) -> None:
        self._state = replace(
            self._state, observation=observation, observation_revision=self._state.observation_revision + 1
        )

    def set_result(self, name: str, result: StoredResult | None) -> None:
        results = dict(self._state.results)
        if result is None:
            results.pop(name, None)
        else:
            results[name] = result
        self._state = replace(self._state, results=MappingProxyType(results))

    def error(self, message: str) -> None:
        self._state = replace(self._state, errors=(*self._state.errors, message))

    def executed(self, action: ExecutedAction, observation: Observation) -> None:
        self.observe(observation)
        grasp = self._state.latest_settled_grasp
        if action.action == "grasp":
            grasp = GraspEvent(action.execution_id, observation.robot.gripper_width_m)
        self._state = replace(
            self._state,
            recent_actions=(*self._state.recent_actions, action)[-self._history_length :],
            latest_settled_grasp=grasp,
        )

    def mark_handled(self, plugin: str, event_id: str) -> None:
        self._state = replace(self._state, handled_events=self._state.handled_events | {(plugin, event_id)})
