"""Build the real context from a small, deterministic simulator substitute."""
from types import SimpleNamespace

import numpy as np

from context import create_context, observe_context


def make_context(settings=None, **fields):
    observation = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0, 0, 0.9]),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "robot0_joint_pos": np.zeros(7),
    }
    executor = SimpleNamespace(
        gripper=-1,
        env=SimpleNamespace(env=SimpleNamespace(workspace_offset=[0, 0, 0.8])),
        camera_frames=lambda: {"agentview": {"rotation": np.eye(3), "image_rotation": 180}},
        eef_frame_world=lambda: np.eye(3),
    )
    context = observe_context(
        create_context() if settings is None else settings,
        executor=executor, observation=observation, client=fields.get("client"), task="task",
    )
    context.update(fields)
    # This substitute has no MuJoCo model; affordance tests provide their own API.
    if context.get("affordance_api") is None:
        context["plugins"] = {**context["plugins"], "affordance": False}
    return context
