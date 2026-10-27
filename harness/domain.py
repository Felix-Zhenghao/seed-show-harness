"""Immutable, simulator-independent observations and execution records."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

from PIL import Image, ImageDraw

Vec3: TypeAlias = tuple[float, float, float]
Matrix3: TypeAlias = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class ImageAsset:
    label: str
    png: bytes
    width: int
    height: int

    @cached_property
    def digest(self) -> str:
        return hashlib.sha256(self.png).hexdigest()

    @cached_property
    def data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")

    @classmethod
    def from_image(cls, label: str, image: Image.Image) -> ImageAsset:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=False)
        return cls(label, buffer.getvalue(), *image.size)

    def annotate(self, point: tuple[int, int]) -> ImageAsset:
        y, x = point
        if any((not isinstance(v, int) or isinstance(v, bool)) or not 0 <= v <= 1000 for v in point):
            raise ValueError("Point must contain grid integers in [0, 1000]")
        pixel_x, pixel_y = round(x * (self.width - 1) / 1000), round(y * (self.height - 1) / 1000)
        image = Image.open(io.BytesIO(self.png)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse((pixel_x - 4, pixel_y - 4, pixel_x + 4, pixel_y + 4), fill="red", outline="white")
        return self.from_image(f"Interaction point on {self.label}", image)


@dataclass(frozen=True)
class CameraFrame:
    name: str
    position_world: Vec3
    rotation_world: Matrix3
    vertical_fov_deg: float
    image_rotation_deg: int
    raw_row_convention: str = "opengl"


@dataclass(frozen=True)
class RobotState:
    """World-frame poses; TCP +Z points toward fingertips and +X along jaw closure."""

    tcp_position: Vec3
    tcp_rotation: Matrix3
    base_rotation: Matrix3
    joint_positions: tuple[float, ...]
    gripper_width_m: float
    gripper_command: float
    base_position: Vec3 | None = None


@dataclass(frozen=True)
class Observation:
    sequence: int
    simulation_time: float
    images: tuple[ImageAsset, ...]
    cameras: tuple[CameraFrame, ...]
    robot: RobotState
    success: bool
    # Optional backend-native vector for physical replay, excluded from prompts.
    simulator_state: tuple[float, ...] | None = None

    def image(self, name: str) -> ImageAsset:
        matches = [image for image in self.images if image.label == name]
        if not matches:
            available = ", ".join(image.label for image in self.images)
            raise ValueError(f"Unknown observation image {name!r}. Available images: {available}")
        if len(matches) != 1:
            raise ValueError(f"Observation image name is ambiguous: {name!r}")
        return matches[0]


@dataclass(frozen=True)
class TaskMetadata:
    environment: str
    suite: str
    task_id: int
    name: str
    instruction: str
    seed: int
    initial_state: int
    robot: str = "Panda"
    simulator_state_format: str | None = None


@dataclass(frozen=True)
class ControlStep:
    index: int
    simulation_time: float
    command: tuple[float, ...]
    robot: RobotState
    success: bool
    simulator_done: bool


@dataclass(frozen=True)
class ExecutionResult:
    observation: Observation
    control_steps: tuple[ControlStep, ...]
    completed: bool


@dataclass(frozen=True)
class ExecutedAction:
    execution_id: str
    decision_id: int
    action: str
    rationale: str
    source: str
    control_steps: int


@dataclass(frozen=True)
class GraspEvent:
    execution_id: str
    measured_width_m: float
