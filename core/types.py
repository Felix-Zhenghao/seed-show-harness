from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    end_reason: str
    video_path: str
    run_dir: str


@dataclass
class Subgoal:
    id: str
    target: str
    affordance: str
    motion: str
    description: str
    completion: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "Subgoal":
        motion = _normalized_motion(data.get("motion"), index)
        affordance = str(data.get("affordance", "")).strip()
        if not affordance:
            raise ValueError(f"Subgoal {index} is missing required affordance")
        description = str(data.get("description") or "").strip()
        if not description:
            raise ValueError(f"Subgoal {index} is missing required description")
        return cls(
            id=str(data.get("id") or f"subgoal_{index}"),
            target=str(data.get("target") or ""),
            affordance=affordance,
            motion=motion,
            description=description,
            completion=str(data.get("completion") or "").strip(),
        )

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "target": self.target,
            "affordance": self.affordance,
            "motion": self.motion,
            "description": self.description,
            "completion": self.completion,
        }


def _normalized_motion(value: Any, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Subgoal {index} is missing required motion")
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").upper()
    if not normalized:
        raise ValueError(f"Subgoal {index} has invalid motion {text!r}")
    return normalized
