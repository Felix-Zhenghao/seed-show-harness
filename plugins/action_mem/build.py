"""Recent completed actions, separate from requests and proposed actions."""
from typing import Any, Mapping

from harness.content import TextBlock
from harness.context import PluginContext
from harness.robot_prompt import describe_actions
from harness.schemas import EMPTY_SCHEMA
from plugins.contracts import PluginResult, PluginSpec


def build(
    arguments: Mapping[str, Any], context: PluginContext, dependencies: Mapping[str, PluginResult]
) -> PluginResult:
    history = describe_actions(context.state.recent_actions)
    return PluginResult({}, (TextBlock(history),) if history else ())


PLUGIN = PluginSpec(
    "action_mem", "every_decision", "decision", "Recent actual executions.", EMPTY_SCHEMA, EMPTY_SCHEMA, build
)
