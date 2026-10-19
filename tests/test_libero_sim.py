"""Opt-in MuJoCo checks on fixed spatial/goal tasks (no VLM cost).

LIBERO_SIM_TESTS=1 third_party/LIBERO-plus/.venv/bin/python -m pytest -q seed_harness/tests/test_libero_sim.py
"""
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("LIBERO_SIM_TESTS") != "1",
                               reason="requires downloaded LIBERO assets and an offscreen GL context")


def test_affordance_and_auto_release_reach_real_runner(tmp_path, monkeypatch):
    """Replay the real example's API selection; exercise physics without paid calls."""
    import json
    import numpy as np
    from types import SimpleNamespace
    from core.sim import libero_runner as runner
    from context import create_context
    from core.sim.libero_config import select_tasks
    from core.sim.libero_tasks import configure_imports
    from core.vlm.vlm_client import VLMResponse

    cfg = create_context()
    cfg["tasks"].update(mode="single", suite="libero_goal", task=0, episodes=1)
    cfg["harness"].update(max_steps=3)
    cfg["plugins"].update(subgoal=False, recovery=False, auto_release=True, affordance=True)
    cfg["semantic_eef_action"]["gripper_steps"] = 40
    selection_calls, controller_prompts = [], []

    class Client:
        def __init__(self, cfg, trace):
            self.session = SimpleNamespace(close=lambda: None)

        def complete_json(self, prompt, frame, **kwargs):
            if "code" in kwargs["schema"]["properties"]:
                selection_calls.append(prompt)
                answer = {"code": "sim.get_object_part(object_name='wooden_cabinet_1', part_name='mesh:wooden_cabinet_middle_handle_vis')",
                          "reasoning": "Replay of the real model's middle-handle selection"}
            else:
                controller_prompts.append(prompt)
                assert "affordance is marked with a RED point" in prompt
                assert np.any(np.all(frame == [255, 0, 0], axis=2))
                answer = {"decision": ("GRASP", "CAM_RIGHT", "MV_UP")[len(controller_prompts) - 1],
                          "reasoning": "Local integration check"}
            return VLMResponse("", json.dumps(answer), {"json": answer})

    monkeypatch.setattr(runner, "LiberoVLMClient", Client)
    configure_imports(cfg, tmp_path)
    runner.run_episode(cfg, select_tasks(cfg)[0], 0, tmp_path)
    records = [json.loads(line) for line in next(tmp_path.rglob("steps.jsonl")).read_text().splitlines()]
    assert len(selection_calls) == 1 and len(controller_prompts) == 3
    assert records[0]["executed_tokens"] == ["GRASP", "RELEASE"]
    assert records[0]["auto_release"] and records[0]["after"]["gripper_width_m"] > 0.07
    assert not records[1]["auto_release"]
    np.testing.assert_allclose(records[1]["before"]["eef_pos"], records[1]["after"]["eef_pos"], atol=1e-8, rtol=0)
    assert records[0]["affordance"]["pixels_xy"]["agentview"] != records[2]["affordance"]["pixels_xy"]["agentview"]
    assert "automatically reopened" in controller_prompts[1]
    assert records[0]["affordance"]["part"]["part_name"] == "mesh:wooden_cabinet_middle_handle_vis"
    assert "Automatic RELEASE" in next(tmp_path.rglob("trajectory.html")).read_text()


def test_mug_vlm_fallback_records_marked_controller_inputs(tmp_path):
    """Real mug scene and trace pipeline, with scripted HTTP replies and no API cost."""
    # LIBERO caches configuration on import; give this scene its own process.
    code = r'''
    import sys
    from pathlib import Path
    import json
    import numpy as np
    from PIL import Image
    from context import create_context
    from core.sim import libero_runner as runner
    from core.sim.libero_config import select_tasks
    from core.sim.libero_tasks import configure_imports
    from core.vlm.vlm_client import VLMClient

    tmp_path = Path(sys.argv[1])
    cfg = create_context()
    cfg["tasks"].update(mode="single", suite="libero_10", task=9, episodes=1)
    cfg["harness"]["max_steps"] = 2
    cfg["plugins"].update(subgoal=False, recovery=False, action_chunk=False)
    cfg["runtime"]["debug"] = True
    cfg["model"]["api_key"] = "scripted-test-key"
    counts = {"selection": 0, "locate": 0, "verify": 0, "controller": 0}

    def post(client, payload):
        prompt = "\n".join(block["text"] for block in payload["messages"][0]["content"]
                           if block["type"] == "text")
        assert "json" in prompt.lower()
        if "MODE: live API selection" in prompt:
            counts["selection"] += 1
            answer = {"code": "None", "reasoning": "The mug rim has no named simulator part"}
        elif "ROLE: AffordancePointer" in prompt:
            counts["locate"] += 1
            answer = {"points": [{"part": "yellow-white mug rim", "why": "Scripted localization",
                                   "point": [400, 600]}]}
        elif "ROLE: AffordanceVerifier" in prompt:
            counts["verify"] += 1
            answer = {"present": True, "on_target": False, "why": "Scripted tracking",
                      "point": [400, 600 + 10 * counts["verify"]]}
        else:
            assert "estimated from the images by a VLM" in prompt
            answer = {"decision": ("CAM_RIGHT", "MV_UP")[counts["controller"]],
                      "reasoning": "Exercise changing camera views"}
            counts["controller"] += 1
        return {"choices": [{"message": {"content": json.dumps(answer)}}]}

    VLMClient._post_chat = post
    configure_imports(cfg, tmp_path)
    result = runner.run_episode(cfg, select_tasks(cfg)[0], 0, tmp_path)
    assert result.steps == 2
    records_path = next(tmp_path.rglob("steps.jsonl"))
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert counts == {"selection": 1, "locate": 2, "verify": 4, "controller": 2}
    assert all(record["affordance"]["source"] == "vlm" for record in records)
    assert records[0]["affordance"]["pixels_xy"] != records[1]["affordance"]["pixels_xy"]
    events = [json.loads(line) for line in (records_path.parent / "trace.jsonl").read_text().splitlines()]
    requests = [e for e in events if e["event"] == "Model input" and e["role"] == "controller"]
    assert len(requests) == len(records) == 2
    for request, record in zip(requests, records):
        assert len(request["images"]) == 2
        for view, path in zip(("agentview", "robot0_eye_in_hand"), request["images"]):
            image = np.asarray(Image.open(records_path.parent / path))
            x, y = record["affordance"]["pixels_xy"][view]
            np.testing.assert_array_equal(image[y, x], [255, 0, 0])
    '''
    result = subprocess.run([sys.executable, "-c", dedent(code), str(tmp_path)],
                            cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.parametrize("variant", ["original", "composed"])
def test_real_single_task(variant, tmp_path):
    # LIBERO and LIBERO-Plus share an import name; isolate them in fresh processes.
    code = r'''
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from context import create_context
from core.sim.libero_config import select_tasks
from core.sim.libero_tasks import configure_imports, create_env, task_assets, language
from core.sim.libero_runner import observation_images
from interpreters.libero_atomic_controller import LiberoAtomicController, displayed_camera_axes

variant, directory, root = sys.argv[1:]
cfg = create_context()
cfg["tasks"].update(mode="single", suite="libero_spatial", task=0, episodes=1, initial_state=0)
if variant == "composed":
    p = cfg["libero"]["perturbations"]
    p.update(language_variant=1, lighting_variant=1, table_texture_variant=1, robot_initial_state=1)
    p["layout"]["distractor_variant"] = 2
    p["camera"]["orbit_yaw_deg"] = 15
    p["sensor_noise"].update(type="gaussian_blur", severity=2)
    cfg["libero"]["settle_steps"] = 0
directory = Path(directory)
configure_imports(cfg, directory)
suite, _, name = select_tasks(cfg)[0]
env, obs, instruction = create_env(cfg, task_assets(cfg, suite, name), directory / "generated", 0)
try:
    if variant == "composed":
        robot = env.robots[0]
        np.testing.assert_allclose(env.sim.data.qpos[robot._ref_joint_pos_indexes], robot.robot_model.init_qpos)
        from core.record.images import prepare_view
        clean = prepare_view(obs["agentview_image"], rotation_degrees=180)
        noisy, _ = observation_images(obs, cfg)
        assert np.mean(np.abs(noisy.astype(float) - clean)) > 0.5
        assert (directory / "generated/scene.xml").is_file()
        assets = task_assets(cfg, suite, name)
        assert instruction == language(assets["language"].read_text())
        assert instruction != language(assets["base"].read_text())
    ctrl = LiberoAtomicController(env, cfg, obs)
    if variant == "original":
        before = obs["robot0_eef_pos"].copy()
        orientation = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
        obs, _, _ = ctrl.execute("MV_UP", obs)
        delta = obs["robot0_eef_pos"] - before
        assert 0.01 < delta[2] < 0.03, delta
        assert np.linalg.norm(delta[:2]) < 0.005, delta
        assert Rotation.from_matrix(ctrl.target_rotation @ orientation.T).magnitude() < 1e-6
        for token, axis, sign in (("RT_ROLL_RIGHT", 0, -1), ("RT_PITCH_FWD", 1, -1), ("RT_YAW_CCW", 2, -1)):
            before = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
            frame = ctrl.eef_frame_world()
            obs, _, _ = ctrl.execute(token, obs)
            after = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
            delta = frame.T @ Rotation.from_matrix(after @ before.T).as_rotvec()
            assert sign * delta[axis] > 0.15, (token, delta)
            assert np.linalg.norm(np.delete(delta, axis)) < 0.1, (token, delta)
        held = ctrl.target_rotation.copy()
        before = obs["robot0_eef_pos"].copy()
        direction = ctrl.eef_frame_world()[:, 2]
        obs, _, _ = ctrl.execute("MV_UP", obs)
        translation = obs["robot0_eef_pos"] - before
        assert 0.01 < translation @ direction < 0.03, translation
        assert np.linalg.norm(translation - (translation @ direction) * direction) < 0.005, translation
        actual = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3)
        assert Rotation.from_matrix(held @ actual.T).magnitude() < 0.06
        # A precise adjustment and the next default-sized action use the measured wrist frame.
        for distance in (0.008, None):
            before = obs["robot0_eef_pos"].copy()
            direction = ctrl.eef_frame_world()[:, 1]
            obs, _, _ = ctrl.execute("MV_RIGHT", obs, step_m=distance)
            expected = cfg["semantic_eef_action"]["step_m"] if distance is None else distance
            np.testing.assert_allclose(ctrl.target_position - before, direction * expected, atol=1e-10)
            achieved = (obs["robot0_eef_pos"] - before) @ direction
            assert 0.4 * expected < achieved < 1.5 * expected, (distance, achieved)
        for angle in (3, None):
            before = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).copy()
            frame = ctrl.eef_frame_world()
            obs, _, _ = ctrl.execute("RT_YAW_CW", obs, angle_deg=angle)
            expected = cfg["semantic_eef_action"]["rotation_step_deg"] if angle is None else angle
            target = frame.T @ Rotation.from_matrix(ctrl.target_rotation @ before.T).as_rotvec()
            np.testing.assert_allclose(target, [0, 0, np.deg2rad(expected)], atol=1e-10)
            after = env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3)
            achieved = (frame.T @ Rotation.from_matrix(after @ before.T).as_rotvec())[2]
            assert 0.4 * np.deg2rad(expected) < achieved < 1.5 * np.deg2rad(expected), (angle, achieved)
    state = env.get_sim_state().copy()
    pixels = obs["agentview_image"].copy()
    obs, commands, _ = ctrl.execute("CAM_YAW_CW", obs)
    idx = ctrl.camera_id
    forward = displayed_camera_axes(env.sim.data.cam_xmat[idx].reshape(3,3), 180)[2]
    position = env.sim.data.cam_xpos[idx].copy()
    obs, _, _ = ctrl.execute("CAM_FWD", obs)
    np.testing.assert_allclose(env.sim.data.cam_xpos[idx] - position, forward * cfg["semantic_eef_action"]["camera_step_m"], atol=1e-8)
    np.testing.assert_array_equal(state, env.get_sim_state())
    assert not commands
    assert np.mean(np.abs(obs["agentview_image"].astype(float) - pixels)) > 1
    print("PASS", variant, instruction)
finally:
    env.close()
'''
    result = subprocess.run([sys.executable, "-c", code, variant, str(tmp_path), str(ROOT)],
                            cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
