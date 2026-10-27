"""Provider-independent model input blocks."""

from dataclasses import dataclass
from typing import TypeAlias

from harness.domain import ImageAsset


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ImageBlock:
    image: ImageAsset
    # Replace this observation view instead of adding a second image to context.
    replaces_observation: str | None = None


ContentBlock: TypeAlias = TextBlock | ImageBlock
