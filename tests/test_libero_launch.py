"""Configuration, frame geometry, model protocol, and task selection regressions."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.action_units import CAMERA_ATOMS, MOVE_ATOMS, RT_ATOMS
from context import create_context
from core.sim.libero_config import perturbed, select_tasks, validate_context
from core.sim.libero_model import DebugTrace, LiberoVLMClient
from core.sim.libero_tasks import language, prepare_task, task_assets
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMResponse
from interpreters.libero_atomic_controller import (
    LiberoAtomicController, move_camera,
)


from tests.context_helpers import make_context


@pytest.fixture
def cfg():
    return create_context()


@pytest.fixture
def catalog(cfg, tmp_path):
    package = tmp_path / "libero/libero"
    (package / "benchmark").mkdir(parents=True)
    (package / "benchmark/libero_suite_task_map.py").write_text(
        "libero_task_map = {'libero_spatial': ['first', 'second'], 'libero_goal': ['goal']}\n")
    cfg["libero"]["original_root"] = str(tmp_path)
    cfg["tasks"].update(mode="single", suite="libero_spatial", task=0)
    return cfg


def test_default_and_selection(catalog):
    assert not perturbed(catalog)
    assert select_tasks(catalog) == [("libero_spatial", 0, "first")]
    catalog["tasks"].update(mode="bundle", task_ids=["second", 0, 0])
    assert select_tasks(catalog) == [("libero_spatial", 1, "second"), ("libero_spatial", 0, "first")]
    catalog["tasks"]["task_ids"] = []
    assert len(select_tasks(catalog)) == 2
    catalog["tasks"].update(mode="benchmark", benchmark_suites=["libero_spatial", "libero_goal"])
    assert len(select_tasks(catalog)) == 3


@pytest.mark.parametrize("bad", [-1, 2, "missing", True])
def test_invalid_task_is_not_wrapped(catalog, bad):
    catalog["tasks"]["task"] = bad
    with pytest.raises(ValueError, match="Unknown task"):
        select_tasks(catalog)


def test_missing_required_field_is_reported(cfg):
    del cfg["semantic_eef_action"]["camera_step_m"]
    with pytest.raises(ValueError, match="camera_step_m"):
        validate_context(cfg)


def test_user_fields_need_no_schema_registration(cfg):
    cfg["object_notes"] = lambda: "cup on the table"
    cfg["semantic_eef_action"]["custom_hint"] = "move gently"
    validate_context(cfg)


@pytest.mark.parametrize("field", ["robot_initial_state", "language_variant", "lighting_variant",
                                   "table_texture_variant", "background_variant"])
def test_each_variant_enables_plus(cfg, field):
    cfg["libero"]["perturbations"][field] = 1
    assert perturbed(cfg)


def test_invalid_combinations(cfg):
    p = cfg["libero"]["perturbations"]
    p["layout"].update(distractor_variant=1, position_level=1)
    with pytest.raises(ValueError, match="distractor_variant OR position_level"):
        validate_context(cfg)
    p["layout"].update(distractor_variant=0, position_level=0)
    p["camera"]["distance_scale"] = 1.005
    with pytest.raises(ValueError, match="0.01 increments"):
        validate_context(cfg)


def test_robot_pose_range_matches_upstream(cfg):
    cfg["libero"]["perturbations"]["robot_initial_state"] = 501
    with pytest.raises(ValueError, match="robot_initial_state"):
        validate_context(cfg)


def test_goal_displacement_uses_moved_bddl_and_matching_state(cfg, tmp_path):
    cfg["libero"].update(original_root=str(tmp_path), plus_root=str(tmp_path))
    cfg["libero"]["perturbations"]["layout"]["position_level"] = 2
    bddls = tmp_path / "libero/libero/bddl_files/libero_goal"
    bddls.mkdir(parents=True)
    (bddls / "goal.bddl").touch()
    moved = bddls / "goal_moved_level2_sample1.bddl"
    moved.touch()
    states = tmp_path / "libero/libero/init_files/libero_newobj/libero_goal/goal_moved_level2_sample1.pruned_init"
    states.parent.mkdir(parents=True)
    states.touch()
    assets = task_assets(cfg, "libero_goal", "goal")
    assert assets["layout"] == moved
    assert assets["initial_states"] == states


def test_composed_task_uses_layout_states_and_language(cfg, tmp_path):
    original, plus = tmp_path / "original", tmp_path / "plus"
    cfg["libero"].update(original_root=str(original), plus_root=str(plus))
    for root in (original, plus):
        (root / "libero/libero/bddl_files/libero_spatial").mkdir(parents=True)
    bddl = '(define (problem LIBERO_Tabletop_Manipulation) (:language original instruction) (:objects bowl))'
    (original / "libero/libero/bddl_files/libero_spatial/first.bddl").write_text(bddl)
    bddls = plus / "libero/libero/bddl_files/libero_spatial"
    (bddls / "first_add_2.bddl").write_text(bddl.replace("bowl)", "bowl distractor)"))
    (bddls / "first_language_3.bddl").write_text(bddl.replace("original instruction", "paraphrased goal"))
    states = plus / "libero/libero/init_files/libero_newobj/libero_spatial/first_add_2.pruned_init"
    states.parent.mkdir(parents=True)
    states.touch()
    p = cfg["libero"]["perturbations"]
    p["layout"]["distractor_variant"] = 2
    p["language_variant"] = 3
    p["camera"].update(orbit_yaw_deg=30, distance_scale=1.25)
    p["robot_initial_state"] = 7
    paths = task_assets(cfg, "libero_spatial", "first")
    path, instruction, kwargs = prepare_task(cfg, paths, tmp_path / "output")
    assert paths["initial_states"] == states
    assert path.endswith("task_view_30_0_125_0_0_initstate_7.bddl")
    assert instruction == "paraphrased goal"
    assert "distractor" in (tmp_path / "output/task.bddl").read_text()
    assert language((tmp_path / "output/task.bddl").read_text()) == instruction
    assert kwargs == {}


def fake_sim(rotation=None, parent_rotation=None):
    rot = np.eye(3) if rotation is None else rotation
    parent = np.eye(3) if parent_rotation is None else parent_rotation
    pos = np.array([0.8, -0.3, 1.5])
    model = SimpleNamespace(cam_bodyid=np.array([0]), cam_pos=np.zeros((1, 3)), cam_quat=np.zeros((1, 4)))
    data = SimpleNamespace(cam_xpos=np.array([pos]), cam_xmat=np.array([rot.flatten()]),
                           body_xmat=np.array([parent.flatten()]), body_xpos=np.array([[0.1, 0.2, 0.3]]))
    def forward():
        data.cam_xpos[0] = parent @ model.cam_pos[0] + data.body_xpos[0]
        data.cam_xmat[0] = (parent @ Rotation.from_quat(model.cam_quat[0][[1, 2, 3, 0]]).as_matrix()).flatten()
    return SimpleNamespace(model=model, data=data, forward=forward)


@pytest.mark.parametrize("token,axis,sign", [
    ("CAM_FWD", 0, 1), ("CAM_BACK", 0, -1), ("CAM_RIGHT", 1, 1),
    ("CAM_LEFT", 1, -1), ("CAM_UP", 2, 1), ("CAM_DOWN", 2, -1),
])
@pytest.mark.parametrize("base_angles", [[0, 0, 0], [20, 30, 40]])
def test_camera_moves_in_base_frame_through_rotated_parent(token, axis, sign, base_angles):
    sim = fake_sim(Rotation.from_euler("xyz", [25, 50, 75], degrees=True).as_matrix(),
                   Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix())
    before = sim.data.cam_xpos[0].copy()
    base = Rotation.from_euler("xyz", base_angles, degrees=True).as_matrix()
    orientation = sim.data.cam_xmat[0].copy()
    move_camera(sim, 0, token, 0.05, 10, base)
    np.testing.assert_allclose(sim.data.cam_xpos[0] - before, sign * base[:, axis] * 0.05, atol=1e-12)
    np.testing.assert_allclose(sim.data.cam_xmat[0], orientation, atol=1e-12)


@pytest.mark.parametrize("token,sign", [("CAM_YAW_CW", 1), ("CAM_YAW_CCW", -1)])
def test_camera_yaw_keeps_position_and_next_move_uses_base_frame(token, sign):
    initial_rotation = Rotation.from_euler("xyz", [25, 50, 75], degrees=True).as_matrix()
    sim = fake_sim(initial_rotation, Rotation.from_euler("x", 20, degrees=True).as_matrix())
    base = Rotation.from_euler("xyz", [10, 30, 50], degrees=True).as_matrix()
    initial_pos = sim.data.cam_xpos[0].copy()
    move_camera(sim, 0, token, 0.05, 90, base)
    expected = Rotation.from_rotvec(sign * base[:, 2] * np.pi / 2).as_matrix() @ initial_rotation
    np.testing.assert_allclose(sim.data.cam_xmat[0].reshape(3, 3), expected, atol=1e-12)
    np.testing.assert_allclose(sim.data.cam_xpos[0], initial_pos, atol=1e-12)
    move_camera(sim, 0, "CAM_FWD", 0.05, 10, base)
    np.testing.assert_allclose(sim.data.cam_xpos[0] - initial_pos, base[:, 0] * 0.05, atol=1e-12)


@pytest.mark.parametrize("token,axis,sign", [
    ("RT_ROLL_LEFT", "x", 1), ("RT_ROLL_RIGHT", "x", -1),
    ("RT_PITCH_FWD", "y", -1), ("RT_PITCH_BACK", "y", 1),
    ("RT_YAW_CW", "z", 1), ("RT_YAW_CCW", "z", -1),
])
@pytest.mark.parametrize("base_angles", [[0, 0, 0], [20, 30, 40]])
def test_eef_uses_controlled_site_and_holds_tilt(cfg, token, axis, sign, base_angles):
    site_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    base = Rotation.from_euler("xyz", base_angles, degrees=True).as_matrix()
    controller = SimpleNamespace(use_delta=True)
    obs = {"robot0_eef_pos": np.array([0.1, 0.2, 1.0]), "robot0_eef_quat": [0, 0, 0, 1]}
    sim = SimpleNamespace(model=SimpleNamespace(camera_name2id=lambda _: 0, body_name2id=lambda _: 0),
                          data=SimpleNamespace(site_xmat=np.array([site_rotation.flatten()]),
                                               body_xmat=np.array([base.flatten()])))
    actions = []
    def step(action):
        actions.append(action.copy())
        obs["robot0_eef_pos"] = action[:3].copy()
        sim.data.site_xmat[0] = Rotation.from_rotvec(action[3:6]).as_matrix().flatten()
        return obs, 0, False, {}
    env = SimpleNamespace(sim=sim, robots=[SimpleNamespace(controller=controller, eef_site_id=0,
                          robot_model=SimpleNamespace(root_body="robot0_base"))],
                          step=step, check_success=lambda: False)
    executor = LiberoAtomicController(env, cfg, obs)
    np.testing.assert_allclose(executor.target_rotation, site_rotation)
    executor.execute(token, obs)
    delta = Rotation.from_euler(axis, sign * cfg["semantic_eef_action"]["rotation_step_deg"], degrees=True).as_matrix()
    target = base @ delta @ base.T @ site_rotation
    np.testing.assert_allclose(executor.target_rotation, target, atol=1e-12)
    expected_position = obs["robot0_eef_pos"] + base[:, 1] * cfg["semantic_eef_action"]["step_m"]
    executor.execute("MV_RIGHT", obs)
    np.testing.assert_allclose(Rotation.from_rotvec(actions[-1][3:6]).as_matrix(), target, atol=1e-12)
    np.testing.assert_allclose(actions[-1][:3], expected_position, atol=1e-12)
    assert all(np.allclose(x, actions[-1]) for x in actions[-cfg["semantic_eef_action"]["sim_steps"]:])
    assert not controller.use_delta


@pytest.mark.parametrize("yaw,sign", [("RT_YAW_CW", 1), ("RT_YAW_CCW", -1)])
@pytest.mark.parametrize("base_angles", [[0, 0, 0], [20, 30, 40]])
def test_wrist_turns_leave_all_translation_and_rotation_axes_fixed_to_base(cfg, yaw, sign, base_angles):
    reference = Rotation.from_euler("xyz", [12, 25, 40], degrees=True).as_matrix()
    base = Rotation.from_euler("xyz", base_angles, degrees=True).as_matrix()
    site = np.array([reference.flatten()])
    obs = {"robot0_eef_pos": np.array([0.1, 0.2, 1.0])}

    def step(action):
        obs["robot0_eef_pos"] = action[:3].copy()
        site[0] = Rotation.from_rotvec(action[3:6]).as_matrix().flatten()
        return obs, 0, False, {}

    env = SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(camera_name2id=lambda _: 0, body_name2id=lambda _: 0),
                            data=SimpleNamespace(site_xmat=site, body_xmat=np.array([base.flatten()]))),
        robots=[SimpleNamespace(controller=SimpleNamespace(use_delta=True), eef_site_id=0,
                                robot_model=SimpleNamespace(root_body="robot0_base"))],
        step=step, check_success=lambda: False,
    )
    executor = LiberoAtomicController(env, cfg, obs)
    np.testing.assert_allclose(executor.base_frame_world(), base, atol=1e-12)
    for _ in range(6):
        executor.execute(yaw, obs)
    yaw_rotation = Rotation.from_rotvec(base[:, 2] * sign * np.pi / 2).as_matrix() @ reference
    np.testing.assert_allclose(executor.target_rotation, yaw_rotation, atol=1e-12)
    np.testing.assert_allclose(executor.base_frame_world(), base, atol=1e-12)

    # Pitch stays around base RIGHT after yaw; rotation composition order matters.
    executor.execute("RT_PITCH_FWD", obs)
    expected_rotation = Rotation.from_rotvec(base[:, 1] * np.deg2rad(-15)).as_matrix() @ yaw_rotation
    np.testing.assert_allclose(executor.target_rotation, expected_rotation, atol=1e-12)
    for token, axis, direction in zip(MOVE_ATOMS, (0, 0, 1, 1, 2, 2), (1, -1, -1, 1, 1, -1)):
        before = obs["robot0_eef_pos"].copy()
        executor.execute(token, obs)
        np.testing.assert_allclose(obs["robot0_eef_pos"] - before, base[:, axis] * direction * 0.02, atol=1e-12)
        np.testing.assert_allclose(executor.target_rotation, expected_rotation, atol=1e-12)


@pytest.mark.parametrize("model", ["deepseek-flash", "kimi-k3"])
@pytest.mark.parametrize("thinking", [False, True])
def test_provider_payload(cfg, tmp_path, model, thinking):
    cfg["model"].update(name=model, api_key="unit-test-key", thinking=thinking)
    client = LiberoVLMClient(cfg, DebugTrace(tmp_path, False))
    request = client._finalize_payload({"model": model, "messages": [], "max_tokens": 345,
                                       "response_format": {"type": "json_object"}})
    if model == "kimi-k3":
        assert request["max_completion_tokens"] == 345
        assert "temperature" not in request and "thinking" not in request
        assert "response_format" not in request
    else:
        assert request["max_tokens"] == 345
        assert request["thinking"] == {"type": "enabled" if thinking else "disabled"}
        assert request["response_format"] == {"type": "json_object"}
        if thinking:
            assert "temperature" not in request
            assert request["reasoning_effort"] == cfg["model"]["reasoning_effort"]
        else:
            assert request["temperature"] == cfg["model"]["temperature"]
    client.session.close()


def test_trace_keeps_exact_images_and_redacts_keys(tmp_path):
    from core.record.images import image_to_data_url
    image = np.full((3, 5, 3), 42, dtype=np.uint8)
    payload = {"model": "deepseek-flash", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "exact prompt"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}]}]}
    trace = DebugTrace(tmp_path, True)
    trace.request(payload)
    trace.event("config", {"api_key": "do-not-leak"})
    from PIL import Image
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "calls/0001_image_0.png")), image)
    request = json.loads((tmp_path / "calls/0001_request.json").read_text())
    assert request["messages"][0]["content"][0]["text"] == "exact prompt"
    assert "do-not-leak" not in (tmp_path / "trajectory.html").read_text()
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:")


def test_extra_actions_reach_prompt_schema_and_parser(cfg):
    class Client:
        def complete_json(self, prompt, image, **kwargs):
            assert all(t in prompt for t in RT_ATOMS + CAMERA_ATOMS)
            assert "CAM_UP" in kwargs["schema"]["properties"]["decision"]["enum"]
            return VLMResponse("", '{"decision":"CAM_UP","reasoning":"inspect"}',
                               {"json": {"decision": "CAM_UP", "reasoning": "inspect"}})
    agent = ControllerAgent(Client())
    response = agent.decide(make_context(cfg))
    assert response.token == "CAM_UP"


def test_model_transport_error_cannot_become_fallback_move(cfg, tmp_path, monkeypatch):
    cfg["model"]["api_key"] = "test-secret"
    client = LiberoVLMClient(cfg, DebugTrace(tmp_path, True))
    def error(_payload):
        raise RuntimeError("HTTP 401: test-secret")
    monkeypatch.setattr("core.vlm.vlm_client.VLMClient._post_chat", lambda self, payload: error(payload))
    with pytest.raises(ConnectionError, match="REDACTED"):
        client._post_chat({"messages": []})
    assert "test-secret" not in (tmp_path / "trajectory.html").read_text()


def test_missing_variant_fails_before_simulation(catalog):
    catalog["libero"]["perturbations"]["language_variant"] = 1
    with pytest.raises(FileNotFoundError, match="Required LIBERO asset"):
        task_assets(catalog, "libero_spatial", "first")


def test_disable_actions_rejected_by_executor(cfg):
    # Covered without touching a simulator: disabled tokens are rejected before writes.
    executor = object.__new__(LiberoAtomicController)
    executor.cfg = copy.deepcopy(cfg)
    executor.cfg["semantic_eef_action"].update(rotation_enabled=False, camera_enabled=False)
    for token in ("CAM_LEFT", "RT_ROLL_LEFT"):
        with pytest.raises(ValueError, match="disabled"):
            executor.execute(token, {})
