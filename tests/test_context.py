"""Context sources, provider lifetime, and extension fields without a simulator."""
from types import SimpleNamespace

import numpy as np
import pytest

from context import create_context, observe_context
from core.sim.libero_config import validate_context
from core.sim.libero_model import write_json


@pytest.fixture
def world():
    observation = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((8, 8, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.9]),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
        "robot0_joint_pos": np.arange(7),
    }
    executor = SimpleNamespace(
        gripper=-1,
        env=SimpleNamespace(env=SimpleNamespace(workspace_offset=[0, 0, 0.8])),
        camera_frames=lambda: {"agentview": {"rotation": np.eye(3), "image_rotation": 180}},
        base_frame_world=lambda: np.eye(3),
    )
    return dict(executor=executor, observation=observation, client=None, task="grasp cup")


def test_settings_are_independent_and_accept_user_fields(tmp_path):
    first, second = create_context(), create_context()
    first["semantic_eef_action"]["step_m"] = 0.03
    first["scene_notes"] = lambda: "cup by plate"
    first["model"]["api_key"] = "private-test-key"
    validate_context(first)
    assert second["semantic_eef_action"]["step_m"] == 0.02
    assert "scene_notes" not in second
    path = tmp_path / "context.json"
    write_json(path, first)
    assert "<callable" in path.read_text()
    assert "private-test-key" not in path.read_text()


def test_live_fields_refresh_without_running_plugin_code(world):
    settings = create_context()
    settings["object_notes"] = lambda: "cup on the left"
    history = ["MV_RIGHT"]
    first = observe_context(settings, **world, subgoal={"motion": "GRASP"}, recent_moves=history)
    assert first["proprioception"] == {
        "eef_pos": [0.1, 0.2, 0.9], "eef_quat_xyzw": [0, 0, 0, 1],
        "gripper_width_m": 0.06, "joint_pos": list(range(7)),
    }
    assert first["table_height_m"] == 0.8
    assert first["base_frame_world"] == np.eye(3).tolist()
    assert first["affordance"] is None and first["plugin_state"] == {}
    assert first["object_notes"] is settings["object_notes"]
    assert "CAM_UP" in first["allowed_tokens"]
    world["observation"]["robot0_eef_pos"][2] = 1.1
    world["executor"].gripper = 1
    history.append("GRASP")
    second = observe_context(settings, **world, subgoal={"motion": "GRASP"}, recent_moves=history)
    assert second["proprioception"]["eef_pos"][2] == 1.1
    assert second["gripper_state"] == "CLOSED" and second["previous_direction"] == "GRASP"
    assert first["recent_moves"] == ["MV_RIGHT"]
    assert first["proprioception"]["eef_pos"][2] == 0.9
    assert "proprioception" not in settings
    assert first["plugin_state"] is second["plugin_state"]


def test_disabled_action_settings_remove_extra_tokens(world):
    settings = create_context()
    settings["semantic_eef_action"].update(rotation_enabled=False, camera_enabled=False)
    settings["libero"]["use_wrist_image"] = False
    current = observe_context(settings, **world)
    assert current["wrist_image"] is None
    assert not any(t.startswith(("CAM_", "RT_")) for t in current["allowed_tokens"])
