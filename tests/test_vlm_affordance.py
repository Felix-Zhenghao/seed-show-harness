"""Exercise the simulator-to-VLM fallback through real builders and controller calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.vlm.roles import ControllerAgent
from plugins.affordance import build as affordance
from tests.context_helpers import make_context


RIM = "left top rim opposite the yellow handle"
STAGE = {"motion": "GRASP", "target": "yellow and white mug", "affordance": RIM}
ABSTAIN = {"code": "None", "reasoning": "Only the whole mug is catalogued; no rim part."}


def located(point, part=RIM):
    return {"points": [{"part": part, "why": "Visible rim edge", "point": point}]}


def verified(point=None, *, present=True):
    return {"present": present, "on_target": point is None,
            "why": "Check the rim edge", "point": point or [0, 0]}


class Client:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, prompt, image, **kwargs):
        assert "json" in prompt.lower()  # Hosted JSON mode requires this in the prompt.
        context = kwargs["context"]
        self.calls.append({"prompt": prompt, "image": image.copy() if image is not None else None,
                           "reference": kwargs.get("wrist_image"),
                           **context["trace"].context})
        assert context["custom_field"] == "preserved"
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(payload={"json": reply})


def fallback_context(client, *, wrist=False):
    events = []
    trace = SimpleNamespace(context={"role": "controller", "step": 0},
                            event=lambda kind, data: events.append((kind, data)))
    api = SimpleNamespace(catalog=lambda: [
        {"object_name": "white_yellow_mug_1", "part_name": "body"},
        {"object_name": "white_yellow_mug_1", "part_name": "mesh:white_yellow_mug_vis"},
    ])
    context = make_context(
        client=client, affordance_api=api, subgoal=STAGE, stage_key="grasp",
        agentview_image=np.zeros((101, 201, 3), dtype=np.uint8),
        wrist_image=np.zeros((201, 101, 3), dtype=np.uint8) if wrist else None,
        trace=trace, custom_field="preserved",
    )
    return context, events


def next_observation(context):
    for field in ("agentview_image", "wrist_image"):
        if context[field] is not None:
            context[field] = np.zeros_like(context[field])
    context["affordance"] = None


def test_mug_rim_fallback_reaches_controller_in_both_views():
    client = Client(ABSTAIN, located([250, 750]), verified([300, 700]),
                    located([700, 300]), verified(),
                    {"decision": "MV_DOWN", "reasoning": "Align with the marked rim"})
    context, events = fallback_context(client, wrist=True)
    clean_front, clean_wrist = context["agentview_image"], context["wrist_image"]
    original_trace = context["trace"].context
    result = ControllerAgent(client).decide(context)
    assert result.token == "MV_DOWN"
    assert context["trace"].context is original_trace
    assert [c.get("operation") for c in client.calls] == [
        "selection", "locate", "verify", "locate", "verify", None,
    ]
    assert not clean_front.any() and not clean_wrist.any()
    # [y,x] on the 0–1000 grid maps to [x,y] pixels in each actual, non-square image.
    record = context["affordance"]
    assert record["source"] == "vlm" and record["query"] == ABSTAIN
    assert record["pixels_xy"] == {"agentview": [140, 30], "robot0_eye_in_hand": [30, 140]}
    np.testing.assert_array_equal(client.calls[2]["image"][25, 150], [255, 0, 0])
    np.testing.assert_array_equal(client.calls[3]["reference"][30, 140], [255, 0, 0])
    assert "SECOND image" in client.calls[3]["prompt"]
    controller = client.calls[-1]
    np.testing.assert_array_equal(controller["image"][30, 140], [255, 0, 0])
    np.testing.assert_array_equal(controller["reference"][140, 30], [255, 0, 0])
    assert "external scene and wrist" in controller["prompt"]
    assert "estimated from the images by a VLM" in controller["prompt"]
    assert "simulator geometry" not in controller["prompt"]
    assert any(kind == "Affordance VLM fallback" for kind, _ in events)
    # Rebuilding or retrying the same observation neither calls the model nor adds dots.
    affordance.build(context)
    assert len(client.calls) == 6


def test_tracks_agentview_drops_hidden_point_and_retries_on_new_observation():
    client = Client(ABSTAIN, located([500, 500]), verified(), verified([300, 750]),
                    verified(present=False), located([400, 250]), verified())
    context, _ = fallback_context(client)
    affordance.build(context)
    next_observation(context)
    affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][30, 150], [255, 0, 0])
    assert not context["agentview_image"][50, 100].any()
    assert client.calls[-1]["operation"] == "track"
    next_observation(context)
    assert "No affordance point" in affordance.build(context)
    assert not context["agentview_image"].any()
    next_observation(context)
    affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][40, 50], [255, 0, 0])
    assert [c["operation"] for c in client.calls].count("selection") == 1
    assert [c["operation"] for c in client.calls].count("locate") == 2


def test_frozen_tracking_is_regrounded_without_a_candidate_dot():
    client = Client(ABSTAIN, located([500, 500]), verified(),
                    verified(), verified(), verified(), located([400, 700]), verified())
    context, _ = fallback_context(client)
    for _ in range(5):
        next_observation(context)
        affordance.build(context)
    assert [c["operation"] for c in client.calls] == [
        "selection", "locate", "verify", "track", "track", "track", "locate", "verify",
    ]
    assert not client.calls[-2]["image"].any()
    np.testing.assert_array_equal(context["agentview_image"][40, 140], [255, 0, 0])


def test_stage_change_discards_previous_contact_and_selects_again():
    client = Client(ABSTAIN, located([500, 500]), verified(),
                    ABSTAIN, located([300, 700], part="microwave shelf"), verified())
    context, _ = fallback_context(client)
    affordance.build(context)
    next_observation(context)
    context.update(stage_key="place", subgoal={"motion": "PLACE", "target": "microwave",
                                               "affordance": "microwave shelf"})
    prompt = affordance.build(context)
    assert "microwave shelf" in prompt and RIM not in prompt
    assert "stage's target" in prompt and "should grasp" not in prompt
    assert [c["operation"] for c in client.calls].count("selection") == 2
    assert client.calls[-2]["operation"] == "locate"
    # Every call family keeps its rules before task, stage, selected part, or view.
    for old, new in zip(client.calls[:3], client.calls[3:]):
        before = old["prompt"].split("Task: ", 1)[0]
        after = new["prompt"].split("Task: ", 1)[0]
        assert before == after
        assert "Return" in before
        if old["operation"] == "selection":
            assert "Available simulator object parts:" in before
            assert "white_yellow_mug_1" in before
        else:
            assert "Coordinates are [y, x]" in before
        assert "microwave shelf" in new["prompt"]
        assert RIM not in new["prompt"]


@pytest.mark.parametrize("reply", [
    {"points": []}, {"points": "bad"}, {"unrelated": 1}, {"points": [None]},
    located([float("nan"), 500]), located([500, float("inf")]),
    located([-1, 500]), located([500, 1001]), located([True, 500]),
    located(["500", 500]), located([500]), located([500, 500], part=""),
    RuntimeError("pointing call failed"), ConnectionError("hosted pointing call failed"),
])
def test_bad_or_missing_points_leave_images_clean_and_do_not_cache_abstention(reply):
    client = Client(ABSTAIN, reply, located([500, 500]), verified())
    context, _ = fallback_context(client)
    original_trace = context["trace"].context
    assert "No affordance point" in affordance.build(context)
    assert not context["agentview_image"].any()
    assert context["trace"].context is original_trace
    next_observation(context)
    affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][50, 100], [255, 0, 0])


@pytest.mark.parametrize("verdict", [
    verified(present=False), verified([1001, 500]),
    {"present": "false", "on_target": True}, {"present": True, "on_target": "true"},
    RuntimeError("verification call failed"), ConnectionError("hosted verification failed"), None,
])
def test_bad_or_hidden_verification_does_not_leak_candidate_dot(verdict):
    client = Client(ABSTAIN, located([500, 500]), verdict)
    context, _ = fallback_context(client)
    assert "No affordance point" in affordance.build(context)
    assert not context["agentview_image"].any()
    assert context["affordance"]["pixels_xy"]["agentview"] is None
    assert context["trace"].context == {"role": "controller", "step": 0}


def test_wrist_can_locate_contact_when_agentview_cannot_see_it():
    client = Client(ABSTAIN, {"points": []}, located([500, 500]), verified())
    context, _ = fallback_context(client, wrist=True)
    prompt = affordance.build(context)
    assert "RED point in the wrist view" in prompt
    assert not context["agentview_image"].any()
    assert client.calls[-1]["reference"] is None
    np.testing.assert_array_equal(context["wrist_image"][100, 50], [255, 0, 0])


@pytest.mark.parametrize("selection", [RuntimeError("selection failed"), ConnectionError("endpoint failed"),
                                       {"code": "sim.step([])", "reasoning": "invalid call"}])
def test_selection_failure_also_uses_visual_fallback(selection):
    client = Client(selection, located([0, 1000]), verified())
    context, _ = fallback_context(client)
    affordance.build(context)
    np.testing.assert_array_equal(context["agentview_image"][0, 200], [255, 0, 0])
    assert context["affordance"]["source"] == "vlm"
