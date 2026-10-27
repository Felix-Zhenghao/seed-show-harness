"""Explain measured robot state using the same directions as the atomic actions."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from harness.domain import ExecutedAction, RobotState

# Canonical TCP: +Z points from wrist to fingertips; +X lies along jaw closure.
# In the reference pose, fingers point down and the jaws close along forward/back.
DOWNWARD_REFERENCE = np.diag([1.0, -1.0, -1.0])


def describe_robot(robot: RobotState) -> str:
    base = np.asarray(robot.base_rotation)
    if robot.base_position is not None:
        position = base.T @ (np.asarray(robot.tcp_position) - robot.base_position)
        position_text = (
            "Gripper center (TCP), measured from the robot base origin: "
            + ", ".join(
                f"{abs(float(value)):.3f} m {positive if value >= 0 else negative}"
                for value, (positive, negative) in zip(
                    position, (("forward", "back"), ("right", "left"), ("up", "down"))
                )
            )
            + "."
        )
    else:
        x, y, z = robot.tcp_position
        position_text = (
            f"Gripper center (TCP), measured from the world origin in metres: "
            f"X={x:.3f}, Y={y:.3f}, Z={z:.3f}. The robot base origin was not supplied."
        )
    rotation = base.T @ np.asarray(robot.tcp_rotation)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always", UserWarning)
        angles = Rotation.from_matrix(rotation @ DOWNWARD_REFERENCE.T).as_euler("xyz", degrees=True)
    roll, pitch, yaw = (0.0 if abs(float(angle)) < 0.05 else float(angle) for angle in angles)
    tilt = float(np.rad2deg(np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0))))
    orientation = (
        f"Gripper orientation: roll {roll:+.1f}°, pitch {pitch:+.1f}°, yaw {yaw:+.1f}°. "
        "Zero means fingers pointing straight down, with the line between the jaws along forward/back. "
        "The angles describe fixed-base roll, then pitch, then yaw. Positive angles follow "
        "rotate roll left, rotate pitch forward, and rotate yaw clockwise, respectively. "
        "A single rotation can change more than one reported angle. "
        f"The fingers currently point {tilt:.1f}° away from straight down."
    )
    if notices:
        orientation += " At this tilt, roll and yaw overlap; the reported angles are one equivalent pose."
    command = "close" if robot.gripper_command > 0 else "open" if robot.gripper_command < 0 else "neutral"
    gripper = f"Gripper opening between fingers: {robot.gripper_width_m * 1000:.1f} mm. Last gripper command: {command}."
    return "\n".join((position_text, orientation, gripper))


def describe_actions(actions: Sequence[ExecutedAction]) -> str:
    if not actions:
        return ""
    lines = ["Previous taken actions and the rationale to take them (most recent first):"]
    lines.extend(f"- {action.action}: {action.rationale}" for action in reversed(actions))
    return "\n".join(lines)
