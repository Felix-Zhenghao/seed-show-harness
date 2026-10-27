"""Semantic actions and pure Cartesian target construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from harness.config import ActionConfig
from harness.domain import Matrix3, RobotState, Vec3
from harness.schemas import object_schema, require_text, validate
from harness.serialization import parse_object


class AtomicAction(str, Enum):
    FORWARD = "move forward"
    BACK = "move back"
    LEFT = "move left"
    RIGHT = "move right"
    UP = "move up"
    DOWN = "move down"
    ROLL_LEFT = "rotate roll left"
    ROLL_RIGHT = "rotate roll right"
    PITCH_FORWARD = "rotate pitch forward"
    PITCH_BACK = "rotate pitch back"
    YAW_CLOCKWISE = "rotate yaw clockwise"
    YAW_COUNTERCLOCKWISE = "rotate yaw counterclockwise"
    GRASP = "grasp"
    RELEASE = "release"
    HOLD = "hold"


TRANSLATIONS: dict[AtomicAction, Vec3] = {
    AtomicAction.FORWARD: (1, 0, 0),
    AtomicAction.BACK: (-1, 0, 0),
    AtomicAction.LEFT: (0, -1, 0),
    AtomicAction.RIGHT: (0, 1, 0),
    AtomicAction.UP: (0, 0, 1),
    AtomicAction.DOWN: (0, 0, -1),
}
# Mathematical rotations use right-handed MuJoCo axes. The reflected displayed
# image convention makes +Z yaw clockwise in a processed top-down image.
ROTATIONS: dict[AtomicAction, Vec3] = {
    AtomicAction.ROLL_LEFT: (1, 0, 0),
    AtomicAction.ROLL_RIGHT: (-1, 0, 0),
    AtomicAction.PITCH_FORWARD: (0, 1, 0),
    AtomicAction.PITCH_BACK: (0, -1, 0),
    AtomicAction.YAW_CLOCKWISE: (0, 0, 1),
    AtomicAction.YAW_COUNTERCLOCKWISE: (0, 0, -1),
}
DECISION_SCHEMA = object_schema(
    {
        "rationale": {"type": "string"},
        "action_choice": {"type": "string", "enum": [action.value for action in AtomicAction]},
    }
)


@dataclass(frozen=True)
class Decision:
    action: AtomicAction
    rationale: str

    @classmethod
    def parse(cls, text: str) -> Decision:
        data = parse_object(text)
        validate(DECISION_SCHEMA, data)
        require_text(data, "rationale", "action_choice")
        return cls(AtomicAction(data["action_choice"]), data["rationale"])


@dataclass(frozen=True)
class CartesianCommand:
    position: Vec3
    rotation: Matrix3
    gripper: float
    control_steps: int

    def vector(self) -> tuple[float, ...]:
        rotation = Rotation.from_matrix(self.rotation).as_rotvec()
        return (*self.position, *(float(v) for v in rotation), self.gripper)


def interpret(action: AtomicAction, robot: RobotState, config: ActionConfig) -> CartesianCommand:
    position = np.array(robot.tcp_position)
    rotation = np.array(robot.tcp_rotation)
    base = np.array(robot.base_rotation)
    gripper = robot.gripper_command
    steps = config.motion_steps
    if action in TRANSLATIONS:
        position += base @ np.array(TRANSLATIONS[action]) * config.translation_m
    elif action in ROTATIONS:
        vector = base @ np.array(ROTATIONS[action]) * np.deg2rad(config.rotation_deg)
        rotation = Rotation.from_rotvec(vector).as_matrix() @ rotation
    elif action in (AtomicAction.GRASP, AtomicAction.RELEASE):
        gripper = 1.0 if action == AtomicAction.GRASP else -1.0
        steps = config.gripper_steps
    elif action == AtomicAction.HOLD:
        steps = 1
    else:
        raise ValueError(f"Unsupported atomic action: {action}")
    return CartesianCommand(
        cast(Vec3, tuple(float(v) for v in position)),
        cast(Matrix3, tuple(tuple(float(v) for v in row) for row in rotation)),
        gripper,
        steps,
    )


def action_description(config: ActionConfig) -> str:
    return (
        "Directions use fixed robot-base axes, independent of wrist orientation: "
        "forward +X, back -X, left -Y, right +Y, up +Z. "
        f"Each translation is {config.translation_m:g} m; each rotation is {config.rotation_deg:g} degrees. "
        "Rotations pivot at the current TCP. Roll left tilts an initially base-up vector toward "
        "base-left; pitch forward tilts it toward base-forward. For yaw, imagine looking "
        "straight down at the robot from above. Yaw clockwise turns the gripper clockwise "
        "around the vertical Z axis; yaw counterclockwise turns it counterclockwise. "
        "Opposite names reverse the motion. Hold maintains the measured pose and existing "
        "gripper effort for one control step. Grasp closes; release opens. "
        "Screen left/right are camera-dependent and are not substitutes for robot-base directions."
    )
