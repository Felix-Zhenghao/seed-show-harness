"""Stable controller instructions generated from the selected capabilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harness.actions import action_description
from harness.config import ActionConfig
from harness.content import TextBlock
from harness.context import PluginContext
from harness.prompting import render_template
from harness.schemas import EMPTY_SCHEMA
from plugins.contracts import PluginResult, PluginSpec


def create_plugin(actions: ActionConfig) -> PluginSpec:
    instructions = render_template(
        Path(__file__).with_name("libero_base_prompt.txt"),
        action_description=action_description(actions),
    )

    def build(
        arguments: Mapping[str, Any], context: PluginContext, dependencies: Mapping[str, PluginResult]
    ) -> PluginResult:
        return PluginResult({}, (TextBlock(instructions),))

    return PluginSpec(
        "base_prompt",
        "every_decision",
        "decision",
        "Atomic action and information-gathering instructions.",
        EMPTY_SCHEMA,
        EMPTY_SCHEMA,
        build,
        presentation="instructions",
    )
