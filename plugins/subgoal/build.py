"""A persistent subgoal produced by a read-only DeepSeek Harness agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from harness.content import TextBlock
from harness.context import PluginContext
from harness.prompting import template
from harness.schemas import PROMPT_SCHEMA, object_schema, require_text, validate_prompt
from harness.serialization import Object
from plugins.contracts import PluginResult, PluginSpec

RESULT_SCHEMA = object_schema({name: {"type": "string"} for name in ("subgoal", "rationale", "completion")})


class PlanningClient(Protocol):
    def plan(self, instructions: str, request: str, context: PluginContext, output_schema: Object) -> Object:
        ...


def validate_result(data: Mapping[str, Any]) -> None:
    require_text(data, "subgoal", "rationale", "completion")


def create_plugin(planner: PlanningClient) -> PluginSpec:
    instructions = template(Path(__file__).with_name("prompt.txt"))

    def build(
        arguments: Mapping[str, Any], context: PluginContext, dependencies: Mapping[str, PluginResult]
    ) -> PluginResult:
        output = planner.plan(instructions, arguments["prompt"], context, RESULT_SCHEMA)
        return PluginResult(
            output,
            (
                TextBlock(
                    f"The current subgoal to complete the task is {output['subgoal']}\n"
                    f"The completion of this subgoal is defined as {output['completion']}"
                ),
            ),
        )

    return PluginSpec(
        "subgoal",
        "model_call",
        "until_next_call",
        "Ask a planning agent to propose or replace the current subgoal using observations and "
        "execution history. Call when planning help is needed or the current subgoal is complete "
        "or unsuitable. Explain the question in prompt. Reuse the result across decisions.",
        PROMPT_SCHEMA,
        RESULT_SCHEMA,
        build,
        validate_arguments=validate_prompt,
        validate_result=validate_result,
    )
