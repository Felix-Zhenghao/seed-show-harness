"""Franka Cartesian pose targets and camera-local active perception in MuJoCo."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import CAMERA_ATOMS, MOVE_ATOMS, RT_ATOMS


LOCAL_MOVE_VECTORS = dict(zip(MOVE_ATOMS, np.array([
    [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1],
], dtype=float)))
# Keep the LIBERO signs, now around wrist-attached FWD/RIGHT/UP axes.
LOCAL_ROTATION_VECTORS = dict(zip(RT_ATOMS, np.array([
    [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1],
], dtype=float)))


def displayed_camera_axes(rotation: np.ndarray, image_rotation: int) -> tuple:
    """World vectors for displayed right/up/forward, including LIBERO's raw upside-down render."""
    angle = np.deg2rad(image_rotation)
    # Raw array columns follow camera +X; raw array rows follow camera +Y.
    # np.rot90 rotates pixels CCW; invert that image transform to get world axes.
    right = np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1]
    up = np.sin(angle) * rotation[:, 0] - np.cos(angle) * rotation[:, 1]
    return right, up, -rotation[:, 2]


def move_camera(sim, camera_id: int, token: str, step_m: float, yaw_deg: float,
                image_rotation: int = 180) -> None:
    """Update camera pose through its parent body, so even non-world parents work."""
    if token not in CAMERA_ATOMS:
        raise ValueError(f"Unknown camera action: {token}")
    model, data = sim.model, sim.data
    position = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    right, up, forward = displayed_camera_axes(rotation, image_rotation)
    directions = dict(zip(CAMERA_ATOMS[:6], (up, -up, forward, -forward, -right, right)))
    if token in directions:
        position += step_m * directions[token]
    else:
        # Rotate optical forward toward displayed right for CW. This remains
        # correct even when the rendered image is mirrored by the camera contract.
        axis = np.cross(forward, right)
        axis /= np.linalg.norm(axis)
        sign = 1 if token == "CAM_YAW_CW" else -1
        rotation = Rotation.from_rotvec(axis * np.deg2rad(yaw_deg) * sign).as_matrix() @ rotation
    parent = model.cam_bodyid[camera_id]
    parent_rotation = data.body_xmat[parent].reshape(3, 3)
    model.cam_pos[camera_id] = parent_rotation.T @ (position - data.body_xpos[parent])
    quat_xyzw = Rotation.from_matrix(parent_rotation.T @ rotation).as_quat()
    model.cam_quat[camera_id] = quat_xyzw[[3, 0, 1, 2]]
    sim.forward()


class LiberoAtomicController:
    def __init__(self, env, cfg: dict, obs: dict):
        self.env, self.cfg = env, cfg
        self.gripper = -1.0
        self.target_position = np.array(obs["robot0_eef_pos"], dtype=float)
        # robot0_eef_quat is the hand BODY frame; OSC controls the grip SITE,
        # which is rotated 90 degrees from it on Panda. Mixing them twists at reset.
        self.target_rotation = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
        # Calibrate FWD/RIGHT/UP to world +X/+Y/+Z at reset, then carry them with the wrist.
        self.reference_rotation = self.target_rotation.copy()
        # Absolute OSC_POSE bypasses normalized delta scaling. Sending a fixed
        # setpoint over N control steps converges to ONE semantic displacement.
        self.env.robots[0].controller.use_delta = False
        self.camera_id = env.sim.model.camera_name2id("agentview")

    def execute(self, token: str, obs: dict, step_m: float | None = None,
                angle_deg: float | None = None) -> tuple[dict, list, bool]:
        a = self.cfg["semantic_eef_action"]
        if token in CAMERA_ATOMS:
            if not a["camera_enabled"]:
                raise ValueError("Camera actions are disabled")
            move_camera(self.env.sim, self.camera_id, token, a["camera_step_m"],
                        a["camera_yaw_step_deg"], self.cfg["libero"]["agentview_rotation"])
            # Render without stepping physics: active perception does not move the arm.
            return self.env.env._get_observations(force_update=True), [], bool(self.env.check_success())
        if token in LOCAL_MOVE_VECTORS:
            direction = self.eef_frame_world() @ LOCAL_MOVE_VECTORS[token]
            self.target_position = np.asarray(obs["robot0_eef_pos"]) + direction * (
                a["step_m"] if step_m is None else step_m)
        elif token in LOCAL_ROTATION_VECTORS:
            if not a["rotation_enabled"]:
                raise ValueError("Rotation actions are disabled")
            rotation = self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id].reshape(3, 3)
            axis = self.eef_frame_world() @ LOCAL_ROTATION_VECTORS[token]
            delta = Rotation.from_rotvec(axis * np.deg2rad(
                a["rotation_step_deg"] if angle_deg is None else angle_deg))
            self.target_position = np.array(obs["robot0_eef_pos"], dtype=float)
            self.target_rotation = delta.as_matrix() @ rotation
        elif token in ("GRASP", "RELEASE"):
            self.gripper = 1.0 if token == "GRASP" else -1.0
        elif token not in ("STOP", "DONE"):
            raise ValueError(f"Unknown semantic action: {token}")
        if token == "DONE":
            return obs, [], bool(self.env.check_success())
        action = np.r_[self.target_position, Rotation.from_matrix(self.target_rotation).as_rotvec(), self.gripper]
        count = a["gripper_steps"] if token in ("GRASP", "RELEASE") else a["sim_steps"]
        commands = []
        success = False
        for _ in range(count):
            obs, _, _, _ = self.env.step(action)
            commands.append(action.tolist())
            success = bool(self.env.check_success())
            if success:
                break
        return obs, commands, success

    def eef_frame_world(self) -> np.ndarray:
        """Measured wrist-attached FWD/RIGHT/UP axes, expressed as world-space columns."""
        rotation = self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id].reshape(3, 3)
        return rotation @ self.reference_rotation.T

    def camera_frames(self) -> dict:
        """Return current geometry for the base plugin's image-direction guidance."""
        frames = {}
        for camera, rotation in (("agentview", self.cfg["libero"]["agentview_rotation"]),
                                 ("robot0_eye_in_hand", self.cfg["libero"]["wrist_rotation"])):
            if camera != "agentview" and not self.cfg["libero"]["use_wrist_image"]:
                continue
            idx = self.env.sim.model.camera_name2id(camera)
            frames[camera] = {
                "rotation": self.env.sim.data.cam_xmat[idx].reshape(3, 3).copy(),
                "image_rotation": rotation,
            }
        return frames
