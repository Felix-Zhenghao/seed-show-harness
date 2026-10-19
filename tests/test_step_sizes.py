"""Per-decision size validation, default selection, execution, and rollout records."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from context import create_context
from core.action_units import MOVE_ATOMS, RT_ATOMS
from core.sim import libero_runner as runner
from core.sim.libero_config import validate_context
from core.step_size import parse_step_size, resolve_step_size
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMParseError, VLMResponse
from interpreters.libero_atomic_controller import LiberoAtomicController
from plugins.action_chunk import build as action_chunk
from plugins.mem_text import build as mem_text
from plugins.recovery.build import RecoveryDecision
from plugins.variable_step import build as variable_step
from tests.context_helpers import make_context


class ScriptedClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts, self.schemas, self.histories = [], [], []
        self.session = SimpleNamespace(close=lambda: None)

    def complete_json(self, prompt, image, **kwargs):
        self.prompts.append(prompt)
        self.schemas.append(kwargs["schema"])
        self.histories.append(list(kwargs["context"]["recent_moves"]))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return VLMResponse("", json.dumps(answer), {"json": answer})

    def complete_token(self, *args, **kwargs):
        # Even a retry transport carrying extra JSON must not reintroduce a size.
        return VLMResponse("MV_FWD", "MV_FWD", {"json": {"distance_m": 0.007}})


@pytest.mark.parametrize("token,field,value", [
    ("MV_RIGHT", "distance_m", 0.008), ("RT_YAW_CCW", "angle_deg", 2),
])
def test_schema_prompt_normalization_and_separate_reasoning(token, field, value):
    answer = {"decision": token, field: value, "reasoning": "Align the gripper.",
              "step_size_reasoning": "Use a smaller step for final alignment."}
    client = ScriptedClient([answer])
    context = make_context()
    context["semantic_eef_action"].update(max_distance_m=0.05, max_angle_deg=30)
    response = ControllerAgent(client).decide(context)
    assert response.payload["json"] == answer
    assert json.loads(response.raw_text) == answer
    assert response.payload["step_size"][field] == value
    assert not response.payload["step_size"]["diagnostics"]
    schema = client.schemas[0]
    assert schema["required"] == ["decision", "reasoning"]
    for size_field, maximum in (("distance_m", 0.05), ("angle_deg", 30)):
        assert schema["properties"][size_field]["type"] == ["number", "null"]
        assert schema["properties"][size_field]["maximum"] == maximum
    assert "Usually omit step sizes" in client.prompts[0]
    assert "[0.0001, 0.05]" in client.prompts[0]
    assert "[0.01, 30]" in client.prompts[0]
    context["recent_moves"] = mem_text.remember(context, [response.token])
    assert mem_text.build_state(context) == f"Recent moves, newest first: {token}"


@pytest.mark.parametrize("token,field", [("MV_LEFT", "distance_m"), ("RT_PITCH_FWD", "angle_deg")])
@pytest.mark.parametrize("value", [0, -0.5, True, False, "0.008", [], {},
                                  float("nan"), float("inf"), -float("inf"), 10**400])
def test_invalid_sizes_fall_back_without_retry(token, field, value, caplog):
    client = ScriptedClient([{"decision": token, field: value, "reasoning": "Align."}])
    response = ControllerAgent(client).decide(make_context())
    assert response.token == token and "retry" not in response.payload
    override = response.payload["step_size"]
    assert field in override["requested"] and override[field] is None
    result = resolve_step_size(token, override, distance_m=0.03, angle_deg=17)
    assert result[f"effective_{field}"] == (0.03 if field == "distance_m" else 17)
    assert result["source"] == "default"
    assert len(result["diagnostics"]) == 1
    assert "using default" in caplog.text


@pytest.mark.parametrize("field,token,low,high", [
    ("distance_m", "MV_FWD", 0.002, 0.06), ("angle_deg", "RT_ROLL_LEFT", 0.5, 30),
])
def test_custom_bounds_are_inclusive_and_reject_outside_values(field, token, low, high):
    actions = create_context()["semantic_eef_action"]
    actions.update({f"min_{field}": low, f"max_{field}": high})
    for value in (low, high):
        assert parse_step_size(token, {field: value}, actions)[field] == value
    for value in (low / 2, high * 2):
        parsed = parse_step_size(token, {field: value}, actions)
        assert parsed[field] is None and parsed["diagnostics"]


@pytest.mark.parametrize("token", ["GRASP", "RELEASE", "DONE", "STOP", "CAM_RIGHT", "CAM_YAW_CW"])
def test_sizes_do_not_apply_to_other_actions(token):
    requested = {"distance_m": 0.008, "angle_deg": 2}
    parsed = parse_step_size(token, requested, create_context()["semantic_eef_action"])
    assert parsed["requested"] == requested
    assert parsed["distance_m"] is None and parsed["angle_deg"] is None
    assert len(parsed["diagnostics"]) == 2


def test_inapplicable_field_does_not_discard_valid_override():
    actions = create_context()["semantic_eef_action"]
    for token, field in (("MV_RIGHT", "distance_m"), ("RT_YAW_CW", "angle_deg")):
        parsed = parse_step_size(token, {"distance_m": 0.008, "angle_deg": 3}, actions)
        assert parsed[field] == (0.008 if field == "distance_m" else 3)
        assert len(parsed["diagnostics"]) == 1


@pytest.mark.parametrize("extra", [{}, {"distance_m": None, "angle_deg": None}])
@pytest.mark.parametrize("token,enabled,wrist,height,expected", [
    ("MV_RIGHT", True, True, 0.82, 0.03),
    ("MV_RIGHT", True, False, 0.82, 0.06),
    ("MV_RIGHT", True, True, 1.0, 0.06),
    ("MV_UP", True, True, 0.82, 0.06),
    ("MV_UP", False, False, 1.0, 0.03),
    ("RT_YAW_CW", True, False, 1.0, 23),
])
def test_omitted_and_null_use_current_defaults(extra, token, enabled, wrist, height, expected):
    context = make_context(token=token, target_in_wrist=wrist)
    context["semantic_eef_action"].update(step_m=0.03, rotation_step_deg=23)
    context["harness"]["coarse_step_m"] = 0.06
    context["plugins"]["variable_step"] = enabled
    context["proprioception"]["eef_pos"][2] = height
    override = parse_step_size(token, extra, context["semantic_eef_action"])
    size = resolve_step_size(token, override, distance_m=variable_step.step_m(context),
                             angle_deg=context["semantic_eef_action"]["rotation_step_deg"])
    field = "distance_m" if token in MOVE_ATOMS else "angle_deg"
    assert size[f"effective_{field}"] == expected
    assert size["source"] == "default" and not size["diagnostics"]
    assert size["requested"] == extra


@pytest.mark.parametrize("extra", [
    {"distance_m": 0.008}, {"distance_m": 0}, {"distance_m": "small"}, {"angle_deg": 3},
])
def test_any_explicit_size_clears_chunk_and_cannot_queue(extra):
    context = make_context()
    context["plugin_state"]["action_chunk"] = ["MV_UP", "MV_DOWN"]
    client = ScriptedClient([{"decision": "MV_RIGHT", **extra,
                              "reasoning": "WRIST: NO; PLAN: MV_RIGHT, MV_UP, MV_DOWN"}])
    response = ControllerAgent(client).decide(context)
    assert response.payload["chunk_plan"] == []
    assert action_chunk.next_action(context) is None
    context.update(token=response.token, chunk_plan=["MV_RIGHT", "MV_UP"])
    action_chunk.queue_plan(context)
    assert action_chunk.next_action(context) is None


def test_new_decisions_reset_sizes_and_null_allows_normal_chunking():
    context = make_context()
    answers = [
        {"decision": "MV_RIGHT", "distance_m": 0.008, "step_size_reasoning": "Align."},
        {"decision": "MV_RIGHT", "distance_m": None, "angle_deg": None,
         "reasoning": "WRIST: NO; PLAN: MV_RIGHT, MV_UP"},
        {"decision": "MV_LEFT", "reasoning": "WRIST: YES"},
    ]
    agent = ControllerAgent(ScriptedClient(answers))
    for index, answer in enumerate(answers):
        response = agent.decide(context)
        assert response.payload["step_size"]["distance_m"] == (0.008 if index == 0 else None)
        context["token"] = response.token
        action_chunk.queue_plan(context)
        assert action_chunk.next_action(context) == ("MV_UP" if index == 1 else None)
    assert "step_size_reasoning" not in response.payload["json"]


def test_size_reasoning_does_not_supply_wrist_or_plan_markers():
    client = ScriptedClient([{"decision": "MV_RIGHT", "reasoning": "", "distance_m": None,
                              "step_size_reasoning": "WRIST: NO; PLAN: MV_RIGHT, MV_UP"}])
    response = ControllerAgent(client).decide(make_context())
    assert response.payload["target_in_wrist"] is None
    assert response.payload["chunk_plan"] == []


@pytest.mark.parametrize("error", [
    VLMParseError("malformed", '{"decision":"MV_RIGHT","distance_m":0.008, '
                  '"reasoning":"WRIST: NO; PLAN: MV_RIGHT, MV_UP"'),
    VLMParseError("malformed", '{"distance_m":0.008'),
])
def test_malformed_recovery_and_token_retries_use_defaults(error):
    context = make_context()
    agent = ControllerAgent(ScriptedClient([
        {"decision": "MV_RIGHT", "distance_m": 0.009}, error,
    ]))
    agent.decide(context)
    response = agent.decide(context)
    assert response.payload["step_size"]["requested"] == {}
    assert response.payload["step_size"]["distance_m"] is None
    assert response.payload["chunk_plan"] == []
    assert "distance_m" not in response.payload["json"]


@pytest.mark.parametrize("bounds", [
    {"min_distance_m": 0}, {"max_distance_m": float("inf")},
    {"min_angle_deg": float("nan")}, {"max_angle_deg": True},
    {"min_distance_m": 0.05, "max_distance_m": 0.01}, {"max_angle_deg": 181},
])
def test_config_rejects_invalid_override_bounds(bounds):
    cfg = create_context()
    cfg["semantic_eef_action"].update(bounds)
    with pytest.raises(ValueError, match="must be a number"):
        validate_context(cfg)


def fake_env(gain=1.0):
    """Small Cartesian plant; gain < 1 models incomplete settling in one control tick."""
    obs = make_context()["observation"]
    obs["robot0_eef_pos"] = np.array([0.1, 0.2, 0.82])
    reference = Rotation.from_euler("xyz", [15, 25, 35], degrees=True).as_matrix()
    data = SimpleNamespace(site_xmat=np.array([reference.flatten()]),
                           body_xmat=np.array([np.eye(3).flatten()]),
                           cam_xpos=np.zeros((2, 3)), cam_xmat=np.tile(np.eye(3).flatten(), (2, 1)))
    commands = []

    def step(command):
        nonlocal obs
        commands.append(command.copy())
        rotation = data.site_xmat[0].reshape(3, 3)
        target = Rotation.from_rotvec(command[3:6]).as_matrix()
        delta = Rotation.from_matrix(target @ rotation.T).as_rotvec() * gain
        data.site_xmat[0] = (Rotation.from_rotvec(delta).as_matrix() @ rotation).flatten()
        obs = {**obs, "robot0_eef_pos": obs["robot0_eef_pos"] +
               gain * (command[:3] - obs["robot0_eef_pos"])}
        return obs, 0, False, {}

    env = SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(camera_name2id=lambda name: int(name != "agentview"),
                                                 body_name2id=lambda _: 0),
                            data=data),
        robots=[SimpleNamespace(controller=SimpleNamespace(use_delta=True), eef_site_id=0,
                                robot_model=SimpleNamespace(root_body="robot0_base"))],
        env=SimpleNamespace(workspace_offset=[0, 0, 0.8]), step=step,
        check_success=lambda: False, close=lambda: None,
    )
    return env, obs, commands


@pytest.mark.parametrize("token,axis,sign", [
    ("RT_ROLL_LEFT", "x", 1), ("RT_ROLL_RIGHT", "x", -1),
    ("RT_PITCH_FWD", "y", -1), ("RT_PITCH_BACK", "y", 1),
    ("RT_YAW_CW", "z", 1), ("RT_YAW_CCW", "z", -1),
])
def test_executor_overrides_follow_base_axes_and_do_not_latch(token, axis, sign):
    cfg = create_context()
    cfg["semantic_eef_action"]["sim_steps"] = 1
    env, obs, commands = fake_env()
    executor = LiberoAtomicController(env, cfg, obs)
    reference = executor.target_rotation.copy()
    obs, _, _ = executor.execute("RT_YAW_CW", obs, angle_deg=90)
    turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    obs, _, _ = executor.execute(token, obs, angle_deg=3)
    rotation = Rotation.from_euler(axis, sign * 3, degrees=True).as_matrix() @ turn @ reference
    np.testing.assert_allclose(executor.target_rotation, rotation, atol=1e-12)
    before = obs["robot0_eef_pos"].copy()
    obs, _, _ = executor.execute("MV_RIGHT", obs, step_m=0.008)
    np.testing.assert_allclose(obs["robot0_eef_pos"] - before, [0, 0.008, 0], atol=1e-12)
    before = obs["robot0_eef_pos"].copy()
    obs, _, _ = executor.execute("MV_LEFT", obs)
    np.testing.assert_allclose(obs["robot0_eef_pos"] - before, [0, -0.02, 0], atol=1e-12)
    obs, _, _ = executor.execute(token, obs)
    rotation = Rotation.from_euler(axis, sign * 15, degrees=True).as_matrix() @ rotation
    np.testing.assert_allclose(executor.target_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(executor.base_frame_world(), np.eye(3), atol=1e-12)
    assert len(commands) == 5
    assert cfg["semantic_eef_action"]["rotation_step_deg"] == 15


def test_runner_sizes_recovery_retry_queue_memory_and_measured_records(tmp_path, monkeypatch):
    cfg = create_context()
    cfg["runtime"]["debug"] = False
    cfg["harness"].update(max_steps=10)
    cfg["semantic_eef_action"].update(sim_steps=1, gripper_steps=1)
    cfg["plugins"].update(subgoal=False, affordance=False, auto_release=False)
    reason = "WRIST: NO; PLAN: MV_RIGHT, MV_UP"
    client = ScriptedClient([
        {"decision": "MV_RIGHT", "distance_m": 0.008, "reasoning": reason,
         "step_size_reasoning": "Use a smaller step for final alignment."},
        {"decision": "MV_LEFT", "distance_m": None, "reasoning": "WRIST: YES"},
        {"decision": "RT_YAW_CCW", "angle_deg": 4, "reasoning": "Align the wrist."},
        {"decision": "RT_YAW_CW", "reasoning": "Turn back."},
        {"decision": "MV_RIGHT", "distance_m": 0, "reasoning": reason},
        {"decision": "INVALID", "distance_m": 0.007, "reasoning": "Unusable action."},
        {"decision": "MV_RIGHT", "distance_m": 0.006, "reasoning": reason},
        {"decision": "MV_RIGHT", "distance_m": None, "reasoning": reason},
    ])
    env, obs, _ = fake_env(gain=0.5)
    monkeypatch.setattr(runner, "LiberoVLMClient", lambda *args: client)
    monkeypatch.setattr(runner, "task_assets", lambda *args: {})
    monkeypatch.setattr(runner, "create_env", lambda *args: (env, obs, "Align the object"))
    before_calls = []

    def recovery(context):
        before_calls.append(context)
        if len(before_calls) == 2:
            return RecoveryDecision("grasp_unsettled", "Recovery adjustment", token="MV_BACK")

    monkeypatch.setattr(runner.recovery, "before_decision", recovery)
    from core.record import episode_logger
    monkeypatch.setattr(episode_logger, "StreamingVideoWriter", lambda *args, **kwargs:
                        SimpleNamespace(frame_count=0, append=lambda frame: None,
                                        close=lambda **kwargs: None))
    result = runner.run_episode(cfg, ("libero_spatial", 0, "task"), 0, tmp_path)
    records = json.loads((Path(result.run_dir) / "steps.json").read_text())
    assert len(records) == 10 and len(client.prompts) == 8
    expected_sizes = [0.008, 0.04, 0.02, 4, 15, 0.04, 0.02, 0.006, 0.04, 0.04]
    for index, (record, expected) in enumerate(zip(records, expected_sizes)):
        size = record["step_size"]
        field = "angle_deg" if record["action"] in RT_ATOMS else "distance_m"
        assert size[f"effective_{field}"] == expected
        assert size["source"] == ("model" if index in (0, 3, 7) else "default")
        assert size["measured"][field] == pytest.approx(expected / 2)
    assert records[0]["vlm"]["c"]["why"] == reason
    assert records[0]["vlm"]["c"]["step_size_reasoning"] == "Use a smaller step for final alignment."
    assert records[1]["action_source"] == "recovery"
    assert records[1]["step_size"]["requested"] == {}
    assert records[2]["step_size"]["requested"] == {"distance_m": None}
    assert records[5]["step_size"]["requested"] == {"distance_m": 0}
    assert records[5]["step_size"]["diagnostics"]
    assert records[6]["step_size"]["requested"] == {}
    assert records[9]["action_source"] == "action_chunk"
    assert records[9]["step_size"]["requested"] == {}
    assert records[9]["vlm"]["c"]["raw"] == ""
    assert client.histories[-1] == [record["action"] for record in records[:8]]
    assert "Recent moves, newest first: MV_BACK, MV_RIGHT" in client.prompts[1]
    assert all(token in MOVE_ATOMS + RT_ATOMS for history in client.histories for token in history)
