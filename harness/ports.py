"""Small injectable interfaces at the simulator and storage boundaries."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from harness.actions import AtomicAction
from harness.domain import ControlStep, ExecutionResult, Observation, TaskMetadata


class EventSink(Protocol):
    def record(self, kind: str, **data: Any) -> None:
        ...


class SimulationEnvironment(Protocol):
    @property
    def metadata(self) -> TaskMetadata:
        ...

    def reset(self) -> Observation:
        ...

    def observe(self) -> Observation:
        ...

    def execute(self, action: AtomicAction, on_step: Callable[[ControlStep], None]) -> ExecutionResult:
        ...

    def close(self) -> None:
        ...
