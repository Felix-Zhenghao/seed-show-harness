"""Standalone builders, context switches, prompt order, and multimodal output."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from context import create_context
from core.prompting.composer import (
    compose, controller_prompt, decision_prompt, planner_prompt, token_prompt, token_retry_prompt,
)
from core.prompting.content import PromptContent, PromptImage, message_blocks, prompt_text
from core.vlm.roles import ControllerAgent
from core.vlm.vlm_client import VLMParseError, VLMResponse, _message_content
from plugins.action_chunk import build as action_chunk
from plugins.auto_release import build as auto_release
from plugins.base import build as base
from plugins.mem_text import build as mem_text
from plugins.recovery import build as recovery
from plugins.subgoal import build as subgoal
from plugins.variable_step import build as variable_step
from plugins.whole_task_stage import build as whole_task_stage
from tests.context_helpers import make_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context():
    return make_context(
        task="Move the cup", extra_context="Keep the cup upright.",
        subgoal={"motion": "GRASP", "target": "cup", "affordance": "rim",
                 "description": "Grasp the cup", "completion": "The cup is held"},
        recent_moves=["MV_UP", "MV_LEFT", "MV_DOWN"],
    )


def test_each_plugin_is_self_contained_and_disabled_before_work(monkeypatch):
    switches = create_context()["plugins"]
    builders = list((ROOT / "plugins").glob("*/build.py"))
    assert {path.parent.name for path in builders} == set(switches)
    for path in builders:
        files = {p.name for p in path.parent.iterdir() if p.is_file()}
        assert files in ({"build.py"}, {"build.py", "prompt.txt"})
        module = importlib.import_module(f"plugins.{path.parent.name}.build")
        # No file reads, world queries, or runtime fields are needed when disabled.
        monkeypatch.setattr(module, "__file__", "/missing/build.py")
        assert module.build({"plugins": {path.parent.name: False}}) == ""


def test_composition_only_calls_builders_and_preserves_finished_content():
    calls = []
    context = {"value": "world"}

    def first(context):
        calls.append(context["value"])
        return '  Finished: ${literal} $$ {"key": 1}  '

    def second(context):
        return "hello " + context["value"]

    assert compose(context, first, second) == '  Finished: ${literal} $$ {"key": 1}  \n\nhello world'
    context["value"] = "updated"
    assert compose(context, second) == "hello updated"
    assert calls == ["world"]


def test_builder_owns_text_and_single_pass_substitution(context, monkeypatch, tmp_path):
    (tmp_path / "prompt.txt").write_text('Task ${task}; {"json":true}; $$5')
    monkeypatch.setattr(base, "__file__", str(tmp_path / "build.py"))
    context["task"] = '${literal} {"x":1}'
    assert base.build(context) == 'Task ${literal} {"x":1}; {"json":true}; $5'
    (tmp_path / "prompt.txt").write_text("Edited: ${task}")
    assert base.build(context).startswith("Edited:")


def test_action_chunk_builds_without_composer(context):
    context["harness"]["action_chunk_steps"] = 7
    assert "plan your next 7 moves" in action_chunk.build(context)
    assert "${step_num}" not in action_chunk.build(context)


def test_base_opens_controller_planner_and_token_retry(context):
    opening = base.build_instructions(context)
    prompt = controller_prompt(context)
    assert prompt.startswith(opening)
    assert planner_prompt(context).startswith(opening)
    assert token_prompt({**context, "previous_prompt": prompt}).startswith(opening)
    assert prompt.count("Your current task:") == prompt.count("Subgoal information:") == 1
    assert "15-degree commands require six steps" in opening
    assert "All MV_*, RT_*, and CAM_* actions use the robot base frame" in opening
    assert "MV_FWD still moves along base +X" in opening
    assert "After any camera yaw, CAM_FWD still moves along base +X" in opening
    assert "looking along base UP" in opening
    assert "CAM_MV_" not in opening and "CAM_RT_" not in opening


@pytest.mark.parametrize("builder", [controller_prompt, planner_prompt])
def test_shared_rules_stay_identical_when_task_stage_and_observations_change(context, builder):
    first = builder(context)
    prefix = first.split("Your current task:\n", 1)[0]
    if builder is controller_prompt:
        assert "WRIST CHECK" in prefix and "ACTION PLAN" in prefix and "Coarse EEF steps" in prefix
        assert "NEVER OSCILLATE" in prefix
    else:
        assert subgoal.build(context) in prefix
    context.update(task="Open the drawer", subgoal={"motion": "PULL", "affordance": "handle"},
                   gripper_state="CLOSED", recovery="auto_release", recent_moves=["GRASP"],
                   agentview_image=np.ones((2, 2, 3), dtype=np.uint8), wrist_image=None)
    context["proprioception"]["eef_pos"][2] = 1.0
    second = builder(context)
    assert second.split("Your current task:\n", 1)[0] == prefix
    assert "Open the drawer" in second and '"affordance": "handle"' in second
    assert '"gripper_state": "CLOSED"' in second and "Move the cup" not in second
    if builder is controller_prompt:
        assert "Recent moves, newest first: GRASP" in second and "20.0 cm above the table" in second
        assert "Holding an object" in second and "automatically reopened" in second


def test_split_builders_remain_standalone_and_honor_switches(context, monkeypatch):
    for module, name in ((base, "base"), (variable_step, "variable_step"), (mem_text, "mem_text")):
        assert module.build_instructions(context) in module.build(context)
        assert module.build_state(context) in module.build(context)
        monkeypatch.setattr(module, "__file__", "/missing/build.py")
        for builder in (module.build, module.build_instructions, module.build_state):
            assert builder({"plugins": {name: False}}) == ""


def test_stage_key_order_does_not_change_prompt_prefix(context):
    first = controller_prompt(context)
    context["subgoal"] = dict(reversed(list(context["subgoal"].items())))
    assert controller_prompt(context) == first


def test_decision_and_token_repairs_keep_shared_prompt_and_camera_order(context):
    shared = controller_prompt(context)
    context["previous_prompt"] = shared
    decision = decision_prompt(context)
    token = token_prompt(context)
    strict = token_retry_prompt({**context, "previous_prompt": token})
    agentview, wrist = (np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20))
    expected_images = message_blocks(PromptContent((PromptImage(agentview), PromptImage(wrist))))
    for prompt in (decision, token, strict):
        blocks = _message_content(prompt, agentview, wrist, image_detail="high", context=context)
        assert [b["type"] for b in blocks] == ["text", "image_url", "image_url"]
        assert blocks[0]["text"].startswith(shared)
        assert [b["image_url"]["url"] for b in blocks[1:]] == [
            b["image_url"]["url"] for b in expected_images
        ]
        assert all(b["image_url"]["detail"] == "high" for b in blocks[1:])
    assert 'Return JSON only: {"decision"' in decision
    assert 'Return JSON only: {"decision"' not in token


def test_base_refreshes_action_sizes_task_and_subgoal(context):
    context["semantic_eef_action"].update(step_m=0.03, rotation_step_deg=30)
    context["task"] = 'Lift the "cup" ${literal}'
    context["subgoal"].update(motion="LIFT", completion="The cup clears the table")
    context["gripper_state"] = "CLOSED"
    prompt = base.build(context)
    assert "Each MV_* normally requests 3 cm; each RT_* requests 30 degrees" in prompt
    assert context["task"] in prompt
    stage = json.loads(prompt.split("Subgoal information:\n", 1)[1])
    assert stage == {**context["subgoal"], "gripper_state": "CLOSED"}


def test_base_handles_disabled_actions_and_missing_wrist():
    settings = create_context()
    settings["semantic_eef_action"].update(rotation_enabled=False, camera_enabled=False)
    settings["libero"]["use_wrist_image"] = False
    context = make_context(settings)
    prompt = base.build(context)
    tokens = prompt.split("Your output action space is:\n", 1)[1].splitlines()[0]
    assert tokens.split(", ") == list(context["allowed_tokens"])
    assert not any(token.startswith(("RT_", "CAM_")) for token in tokens.split(", "))
    assert "If only one camera view is supplied, it is Agentview" in prompt
    assert "GRASP only when the available views" in prompt
    assert json.loads(prompt.split("Subgoal information:\n", 1)[1]) == {"gripper_state": "OPEN"}


def test_controller_refreshes_history_and_height(context):
    context["harness"]["mem_text_len"] = 2
    first = controller_prompt(context)
    assert "Recent moves, newest first: MV_DOWN, MV_LEFT" in first
    assert "10.0 cm above the table" in first and "approach the target's height" in first
    assert first.count("WRIST CHECK") == 1
    context["proprioception"]["eef_pos"][2] = 1.0
    context["gripper_state"] = "CLOSED"
    context["recent_moves"].append("GRASP")
    second = controller_prompt(context)
    assert "20.0 cm above the table" in second and "Holding an object" in second
    assert "approach the target's height" not in second
    assert "Recent moves, newest first: GRASP, MV_DOWN" in second


def test_only_enabled_plugins_contribute(context):
    context["plugins"] = {name: False for name in context["plugins"]}
    assert controller_prompt(context) == ""
    context["plugins"].update(base=True)
    result = controller_prompt(context)
    assert "Your current task:\nMove the cup" in result and '"completion": "The cup is held"' in result
    assert "WRIST CHECK" not in result and "Measured robot" not in result


def test_wrist_switch_is_independent_of_other_plugins(context):
    context["plugins"].update(variable_step=False, action_chunk=False)
    assert controller_prompt(context).count("WRIST CHECK") == 1
    context["plugins"]["wrist_visibility"] = False
    assert "WRIST CHECK" not in controller_prompt(context)


def test_recovery_and_auto_release_own_their_feedback(context):
    context["recovery"] = recovery.RecoveryDecision("unverified_grasp", "sensor", grasp_empty=True)
    assert recovery.build(context).startswith("Recovery: Empty close")
    context["plugins"]["recovery"] = False
    context["recovery"] = "auto_release"
    assert recovery.build(context) == ""
    assert auto_release.build(context).startswith("Empty closed gripper automatically reopened")


def test_memory_recording_lives_in_plugin(context):
    context["harness"]["mem_text_len"] = 2
    assert mem_text.remember(context, ["GRASP", "RELEASE"]) == ["GRASP", "RELEASE"]
    context["plugins"]["mem_text"] = False
    assert mem_text.remember(context, ["GRASP"]) == []


def test_controller_retry_preserves_context_without_json_contract(context):
    calls = []

    class Client:
        def complete_json(self, prompt, image, **kwargs):
            calls.append(prompt)
            raise VLMParseError("VLM returned non-JSON", "invalid")

        def complete_token(self, prompt, tokens, image, **kwargs):
            calls.append(prompt)
            return VLMResponse("MV_UP", "MV_UP", {})

    context.update(task="Move ${task}", extra_context="Return JSON only in another task.\nKEEP THIS LINE.")
    agent = ControllerAgent(Client())
    result = agent.decide(context)
    assert result.token == "MV_UP"
    assert agent.last_prompt == calls[0]
    for prompt in calls:
        assert "KEEP THIS LINE." in prompt and "Move ${task}" in prompt and "The cup is held" in prompt
    assert 'Return JSON only: {"decision"' in calls[0]
    assert 'Return JSON only: {"decision"' not in calls[1]
    assert "exactly one allowed token" in calls[1]


def test_new_image_plugin_composes_without_registration(context, tmp_path, monkeypatch):
    (tmp_path / "history.py").write_text(
        'from core.prompting.content import PromptContent, PromptImage\n'
        'def build(context):\n'
        '    if not context["plugins"].get("history"):\n        return ""\n'
        '    return PromptContent(("Execution history:", PromptImage(context["history_image"]), "Continue."))\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from history import build

    context["plugins"]["history"] = True
    context["history_image"] = np.full((2, 2, 3), 10, dtype=np.uint8)
    content = compose(context, build, base.build)
    retry = token_prompt({**context, "previous_prompt": content})
    blocks = _message_content(retry, np.ones((2, 2, 3), dtype=np.uint8), context=context)
    assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url"]
    assert "Execution history:" in blocks[0]["text"]
    assert "Continue." in blocks[2]["text"] and "Your current task:\nMove the cup" in blocks[2]["text"]
    assert blocks[1]["image_url"] != blocks[3]["image_url"]


def test_image_slots_preserve_order_and_detail():
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20)]
    content = PromptContent(("before", PromptImage(images[0], "high"), "between", PromptImage(images[1])))
    assert prompt_text(content) == "before\n\nbetween"
    blocks = message_blocks(content)
    assert [block["type"] for block in blocks] == ["text", "image_url", "text", "image_url"]
    assert blocks[1]["image_url"]["detail"] == "high"
    assert "detail" not in blocks[3]["image_url"]


def test_planner_fallback_preserves_task_data(context):
    def invalid(*args, **kwargs):
        raise RuntimeError("VLM returned empty content")

    from core.prompting.composer import planner_prompt
    context.update(task='move "cup" ${literal}', client=SimpleNamespace(complete_json=invalid, complete_text=invalid))
    context["planner_prompt"] = planner_prompt(context)
    context["fallback_stage"] = json.loads(whole_task_stage.build({**context, "fallback": True}))
    stages, raw = subgoal.plan(context)
    assert stages[0].id == "task_fallback"
    assert stages[0].description == 'Complete the task directly: move "cup" ${literal}'
    assert len(context["planner_diagnostics"]["errors"]) == 4
    assert json.loads(raw)["subgoals"][0] == stages[0].to_prompt_dict()
