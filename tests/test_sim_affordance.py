"""Simulator part selection, live marker tracking, and independent auto-release."""
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.affordance import build as affordance
from plugins.affordance.build import execute_part_query
from plugins.auto_release import build as auto_release
from plugins.recovery import build as recovery
from plugins.action_chunk import build as action_chunk
from plugins.variable_step import build as variable_step
from core.vlm.vlm_client import VLMResponse
from tests.context_helpers import make_context


class API:
    x = 12
    lookups = 0

    def catalog(self):
        return [{"object_name": "cabinet", "part_name": "middle_handle"}]

    def get_object_part(self, object_name, part_name):
        self.lookups += 1
        if (object_name, part_name) != ("cabinet", "middle_handle"):
            raise ValueError("Unknown simulator object part")
        return {"object_name": object_name, "part_name": part_name, "world_position": [self.x, 15, 1]}

    def project(self, position, camera_name, image_shape):
        return position[:2] if position[0] < image_shape[1] else None


QUERY = "sim.get_object_part(object_name='cabinet', part_name='middle_handle')"
STAGE = {"motion": "GRASP", "target": "cabinet", "affordance": "middle drawer handle"}


@pytest.mark.parametrize("code", [
    "__import__('os').system('true')", "sim.step([1])",
    "sim.get_object_part(object_name='cabinet', part_name=str('middle_handle'))",
    "sim.get_object_part(object_name='cabinet', part_name='middle_handle', x=1)",
    "sim.get_object_part(**{'object_name': 'cabinet', 'part_name': 'middle_handle'})",
    "sim.get_object_part(object_name='cabinet', part_name='middle_handle'); sim.step([])",
])
def test_code_cannot_call_other_functions(code):
    api = API()
    with pytest.raises(ValueError):
        execute_part_query(api, code)
    assert api.lookups == 0


def test_only_catalogued_parts_and_explicit_abstention():
    api = API()
    assert execute_part_query(api, "None") is None
    assert api.lookups == 0
    with pytest.raises(ValueError, match="Unknown"):
        execute_part_query(api, QUERY.replace("middle_handle", "invented_handle"))


def test_marker_follows_part_without_more_model_calls_and_drops_offscreen():
    api, calls = API(), []

    def complete(prompt, image, **kwargs):
        calls.append(prompt)
        assert kwargs["context"]["object_notes"] == "custom scene data"
        return SimpleNamespace(payload={"json": {"code": QUERY, "reasoning": "Handle for grasping"}})

    client = SimpleNamespace(complete_json=complete)
    clean = np.zeros((40, 40, 3), dtype=np.uint8)
    context = make_context(client=client, affordance_api=api, subgoal=STAGE,
                           agentview_image=clean, wrist_image=clean, object_notes="custom scene data")
    prompt = affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][15, 12], [255, 0, 0])
    np.testing.assert_array_equal(context["wrist_image"][15, 12], [255, 0, 0])
    assert not clean.any() and "should grasp" in prompt
    lookups = api.lookups
    assert affordance.build(context) == prompt
    assert len(calls) == 1 and api.lookups == lookups
    api.x = 24
    context.update(agentview_image=clean, wrist_image=clean, affordance=None)
    affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][15, 24], [255, 0, 0])
    assert not context["agentview_image"][15, 12].any()
    api.x = 100
    context.update(agentview_image=clean, wrist_image=clean, affordance=None)
    assert "no red point" in affordance.build(context)
    assert not context["agentview_image"].any()
    assert context["affordance"]["pixels_xy"]["agentview"] is None
    assert len(calls) == 1
    # New episodes do not inherit the selected part from a previous episode.
    fresh = make_context(client=client, affordance_api=api, subgoal=STAGE,
                         agentview_image=clean, wrist_image=clean, object_notes="custom scene data")
    affordance.build(fresh)
    assert len(calls) == 2


def test_disabled_or_nonspatial_affordance_never_calls_model():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    for enabled, motion in ((False, "GRASP"), (True, "LIFT")):
        context = make_context(subgoal={**STAGE, "motion": motion}, agentview_image=frame)
        context["plugins"]["affordance"] = enabled
        assert affordance.build(context) == ""
        assert context["agentview_image"] is frame and context["affordance"] is None


@pytest.mark.parametrize("enabled,closed,width,success,expected", [
    (True, True, 0.001, False, True), (False, True, 0.001, False, False),
    (True, False, 0.001, False, False), (True, True, 0.025, False, False),
    (True, True, 0.001, True, False),
])
def test_auto_release_independent_of_planner_and_recovery(enabled, closed, width, success, expected):
    actions = []

    def execute(token, current):
        actions.append(token)
        return current, [[-1]], False

    context = make_context(success=success, gripper_state="CLOSED" if closed else "OPEN",
                           executor=SimpleNamespace(execute=execute))
    context["plugins"].update(auto_release=enabled, recovery=False, subgoal=False)
    context["proprioception"]["gripper_width_m"] = width
    result = auto_release.after_step(context)
    assert result[3] is expected
    assert actions == (["RELEASE"] if expected else [])


def test_recovery_does_not_reopen_when_auto_release_disabled():
    context = make_context(subgoals=[SimpleNamespace(motion="GRASP")], gripper_state="CLOSED",
                           token="DONE", subgoal_done=True)
    context["plugins"]["auto_release"] = False
    context["proprioception"]["gripper_width_m"] = 0.001
    assert recovery.before_decision(context) is None
    result = recovery.after_step(context)
    assert result.block_done and not result.release


def test_unsettled_grasp_holds_are_bounded_and_state_is_per_context():
    context = make_context(subgoals=[SimpleNamespace(motion="GRASP")], gripper_state="CLOSED")
    context["proprioception"]["gripper_width_m"] = 0.08
    assert [recovery.before_decision(context).token for _ in range(3)] == ["STOP"] * 3
    assert recovery.before_decision(context) is None
    other = make_context(subgoals=context["subgoals"], gripper_state="CLOSED")
    assert recovery.before_decision(other).token == "STOP"
    assert other["plugin_state"] is not context["plugin_state"]


def test_action_chunk_and_variable_step_share_wrist_result_through_context():
    context = make_context(target_in_wrist=False, token="MV_RIGHT")
    context["response"] = VLMResponse("MV_RIGHT", "", {
        "json": {"reasoning": "PLAN: MV_RIGHT, MV_NOPE, MV_UP, MV_DOWN, MV_FWD"},
    })
    action_chunk.read_response(context)
    assert context["chunk_plan"] == ["MV_RIGHT", "MV_UP", "MV_DOWN"]
    action_chunk.queue_plan(context)
    assert action_chunk.next_action(context) == "MV_UP"
    assert variable_step.step_m(context) == context["harness"]["coarse_step_m"]
    context["plugins"].update(action_chunk=False, variable_step=False)
    assert action_chunk.next_action(context) is None
    assert variable_step.step_m(context) == context["semantic_eef_action"]["step_m"]
