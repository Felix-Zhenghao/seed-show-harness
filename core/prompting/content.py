"""Ordered text and image content, independent of model providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptImage:
    image: Any
    detail: str | None = None


@dataclass(frozen=True)
class PromptContent:
    parts: tuple[str | PromptImage, ...]


Prompt = str | PromptContent


def join_content(chunks: Iterable[Prompt], separator: str = "\n\n") -> Prompt:
    """Append chunks, merging adjacent text while preserving every image's position."""
    parts: list[str | PromptImage] = []
    for chunk in chunks:
        if chunk == "":
            continue
        if parts and separator:
            parts.append(separator)
        parts.extend(chunk.parts if isinstance(chunk, PromptContent) else (chunk,))
    merged: list[str | PromptImage] = []
    for part in parts:
        if isinstance(part, str) and merged and isinstance(merged[-1], str):
            merged[-1] += part
        elif not isinstance(part, str) or part.strip():
            merged.append(part)
    if all(isinstance(part, str) for part in merged):
        return "".join(merged)
    return PromptContent(tuple(merged))


def prompt_text(prompt: Prompt) -> str:
    """Text for human-readable logs; request tracing records the images separately."""
    if isinstance(prompt, str):
        return prompt
    return "\n\n".join(part.strip() for part in prompt.parts if isinstance(part, str) and part.strip())


def message_blocks(prompt: Prompt) -> list[dict]:
    from core.record.images import image_to_data_url

    parts = prompt.parts if isinstance(prompt, PromptContent) else (prompt,)
    blocks = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                blocks.append({"type": "text", "text": part.strip()})
        else:
            image = {"url": image_to_data_url(part.image)}
            if part.detail:
                image["detail"] = part.detail
            blocks.append({"type": "image_url", "image_url": image})
    return blocks
