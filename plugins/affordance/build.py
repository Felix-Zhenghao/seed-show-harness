"""Decision-scoped interaction-point localization with a subgoal prerequisite."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from harness.content import ImageBlock, TextBlock
from harness.context import PluginContext
from harness.models.deepseek import CompletionClient
from harness.prompting import observation_blocks, task_input, template, wire_blocks
from harness.schemas import object_schema, require_text, validate
from harness.serialization import Object, parse_object
from plugins.contracts import Dependency, PluginResult, PluginSpec

ARGUMENT_SCHEMA = object_schema(
    {
        "prompt": {
            "type": "string",
            "description": "According to the current subgoal, describe a object part or location in the scene that is suitable for interaction. Write a prompt for the annotator to locate the point.",
        },
        "image_name": {
            "type": "string",
            "description": "Exact label of ONE current observation image to annotate. Select any available view, "
            "such as agentview (external scene) or robot0_eye_in_hand (wrist view). All current views are "
            "provided as context, but the point is only annotated on the selected view.",
        },
    }
)

RESULT_SCHEMA = object_schema(
    {
        "rationale": {"type": "string"},
        "point": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
    }
)
# Responses requires an object at the root; unions belong inside a property.
HELPER_SCHEMA = object_schema(
    {"result": {"anyOf": [RESULT_SCHEMA, object_schema({"error": {"type": "string"}})]}}
)


def validate_arguments(data: Mapping[str, Any]) -> None:
    require_text(data, "prompt", "image_name")


def validate_result(data: Mapping[str, Any]) -> None:
    require_text(data, "rationale")
    point = data["point"]
    if (
        not isinstance(point, (tuple, list))
        or len(point) != 2
        or any(
            (not isinstance(value, int) or isinstance(value, bool)) or not 0 <= value <= 1000
            for value in point
        )
    ):
        raise ValueError("point must contain exactly two integers [y, x] in [0, 1000]")


def planning_arguments(arguments: Mapping[str, Any], context: PluginContext) -> Object:
    image = context.state.observation.image(arguments["image_name"])
    return {
        "prompt": f"Establish a suitable subgoal for this interaction-point request in image {image.label!r}: "
        + arguments["prompt"]
    }


def create_plugin(client: CompletionClient) -> PluginSpec:
    instructions = template(Path(__file__).with_name("prompt.txt"))

    def build(
        arguments: Mapping[str, Any], context: PluginContext, dependencies: Mapping[str, PluginResult]
    ) -> PluginResult:
        state = context.state
        target_image = state.observation.image(arguments["image_name"])
        reply = client.complete(
            role="affordance",
            instructions=instructions,
            output_schema=HELPER_SCHEMA,
            input_items=[
                task_input(state.task.instruction),
                {"role": "user", "content": wire_blocks(dependencies["subgoal"].context_blocks)},
                {"role": "user", "content": wire_blocks(observation_blocks(state.observation))},
                {
                    "role": "user",
                    "content": f"The requested view to annotate (image_name) is {target_image.label}.\n"
                    f"Localization request: {arguments['prompt']}",
                },
            ],
            decision_id=state.decision_id,
        )
        if reply.tool_calls:
            raise ValueError("Affordance helper unexpectedly called a tool")
        envelope = parse_object(reply.text)
        validate(HELPER_SCHEMA, envelope)
        output = envelope["result"]
        if "error" in output:
            raise RuntimeError(f"Affordance helper failed: {output['error']}")
        validate(RESULT_SCHEMA, output)
        validate_result(output)
        point = (output["point"][0], output["point"][1])
        annotated = target_image.annotate(point)
        return PluginResult(
            output,
            (
                TextBlock(
                    f"Suggested interaction point in {target_image.label}: y={point[0]}, x={point[1]} "
                    f"on its 0–1000 image grid. {output['rationale']}"
                ),
                ImageBlock(annotated, replaces_observation=target_image.label),
            ),
        )

    return PluginSpec(
        "affordance",
        "model_call",
        "decision",
        "Annotate a suggestive interaction point on the selected image according to the current subgoal with a red dot to guide action generation.",
        ARGUMENT_SCHEMA,
        RESULT_SCHEMA,
        build,
        (Dependency("subgoal", planning_arguments),),
        validate_arguments=validate_arguments,
        validate_result=validate_result,
    )
