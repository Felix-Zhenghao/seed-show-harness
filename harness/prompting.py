"""Explicit prompt assembly: active results and historical protocol are distinct."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from harness.content import ContentBlock, ImageBlock, TextBlock
from harness.context import EpisodeContext
from harness.domain import Observation
from harness.robot_prompt import describe_robot
from harness.serialization import Object
from plugins.contracts import PluginResult, PluginSpec


@lru_cache(maxsize=32)
def template(path: Path) -> str:
    return path.read_text().strip()


def render_template(path: Path, **values: str) -> str:
    """Only named placeholders are substituted; JSON braces stay literal."""
    text = template(path)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def wire_blocks(blocks: Sequence[ContentBlock]) -> list[Object]:
    result: list[Object] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            result.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageBlock):
            result.extend(
                [
                    {"type": "input_text", "text": block.replaces_observation or block.image.label},
                    {"type": "input_image", "image_url": block.image.data_url, "detail": "original"},
                ]
            )
    return result


def observation_blocks(observation: Observation) -> tuple[ContentBlock, ...]:
    return tuple(ImageBlock(image) for image in observation.images)


def supplemental_blocks(blocks: Sequence[ContentBlock]) -> tuple[ContentBlock, ...]:
    """Replacement images belong in observation slots, not in result snapshots."""
    return tuple(
        block for block in blocks if not isinstance(block, ImageBlock) or block.replaces_observation is None
    )


def current_observation_input(state: EpisodeContext) -> Object:
    """Send each current view once, applying active plugin replacements by label."""
    replacements: dict[str, ImageBlock] = {}
    for record in state.results.values():
        if record.observation_revision != state.observation_revision:
            continue  # Never apply an annotation to a newer camera frame.
        for block in record.result.context_blocks:
            if isinstance(block, ImageBlock) and block.replaces_observation is not None:
                replacements[block.replaces_observation] = block
    blocks = tuple(replacements.get(image.label, ImageBlock(image)) for image in state.observation.images)
    return {"role": "user", "content": wire_blocks(blocks)}


def controller_instructions(state: EpisodeContext, registry: Mapping[str, PluginSpec]) -> str:
    instructions: list[str] = []
    for name, record in sorted(state.results.items()):
        if registry[name].presentation == "instructions":
            instructions.extend(
                block.text for block in record.result.context_blocks if isinstance(block, TextBlock)
            )
    return "\n\n".join(instructions)


def task_input(instruction: str) -> Object:
    return {"role": "user", "content": [{"type": "input_text", "text": "Task: " + instruction}]}


def decision_prefix(
    state: EpisodeContext, registry: Mapping[str, PluginSpec], presented: frozenset[str] = frozenset()
) -> list[Object]:
    """Present the task and current persistent guidance once, ahead of changing observations."""
    # Keep stable task and long-lived context ahead of the current images.
    # Observation images are assembled separately for every model request.
    items = [task_input(state.task.instruction)]
    for name, record in sorted(state.results.items()):
        if (
            name not in presented
            and registry[name].lifetime == "until_next_call"
            and registry[name].presentation == "context"
        ):
            blocks = supplemental_blocks(record.result.context_blocks)
            if blocks:
                items.append({"role": "user", "content": wire_blocks(blocks)})
    return items


def current_context(
    state: EpisodeContext, registry: Mapping[str, PluginSpec], presented: frozenset[str] = frozenset()
) -> Object:
    blocks: list[ContentBlock] = [
        TextBlock(
            "Below is the current state that can help you choose the next action.\n"
            + describe_robot(state.observation.robot)
        )
    ]
    for name, record in sorted(state.results.items()):
        if (
            name in presented
            or registry[name].presentation == "instructions"
            or registry[name].lifetime == "until_next_call"
        ):
            continue
        blocks.extend(supplemental_blocks(record.result.context_blocks))
    return {"role": "user", "content": wire_blocks(blocks)}


def tool_result_output(result: PluginResult) -> str | list[Object]:
    """Use plugin-authored guidance in tool messages; structured data belongs in the log."""
    blocks = supplemental_blocks(result.context_blocks)
    if not blocks:
        return "The tool returned no additional guidance."
    if all(isinstance(block, TextBlock) for block in blocks):
        return "\n\n".join(block.text for block in blocks if isinstance(block, TextBlock))
    return wire_blocks(blocks)
