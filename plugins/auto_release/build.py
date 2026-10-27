"""Once-per-grasp conditional intervention, separate from information tools."""
from pathlib import Path
from typing import Any, Callable, Mapping

from harness.context import PluginContext
from harness.prompting import template
from harness.schemas import EMPTY_SCHEMA, object_schema
from plugins.contracts import PluginResult, PluginSpec


def create_plugin(release: Callable[[str], None], empty_width_m: float) -> PluginSpec:
    rationale = template(Path(__file__).with_name("prompt.txt"))

    def condition(context: PluginContext) -> bool:
        state = context.state
        event = state.latest_settled_grasp
        return (
            event is not None
            and ("auto_release", event.execution_id) not in state.handled_events
            and state.observation.robot.gripper_command > 0
            and state.observation.robot.gripper_width_m < empty_width_m
            and not state.observation.success
        )

    def build(
        arguments: Mapping[str, Any], context: PluginContext, dependencies: Mapping[str, PluginResult]
    ) -> PluginResult:
        event = context.state.latest_settled_grasp
        if event is None:
            raise RuntimeError("No settled grasp event")
        width = context.state.observation.robot.gripper_width_m
        release(rationale)  # Records the explanation with the executed action and fresh observations.
        return PluginResult(
            {"event_id": event.execution_id, "measured_width_m": width, "intervention": "release"},
            observations_changed=True,
            handled_event_id=event.execution_id,
        )

    return PluginSpec(
        "auto_release",
        "conditional",
        "decision",
        "Reopen an apparently empty gripper after settled grasp feedback.",
        EMPTY_SCHEMA,
        object_schema(
            {
                "event_id": {"type": "string"},
                "measured_width_m": {"type": "number"},
                "intervention": {"type": "string"},
            }
        ),
        build,
        condition=condition,
    )
