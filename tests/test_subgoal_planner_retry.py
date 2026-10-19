"""Planner retries and stage merging live inside the subgoal plugin."""
import json
from pathlib import Path
from types import SimpleNamespace

from core.prompting.composer import planner_prompt
from plugins.subgoal import build as subgoal
from plugins.base import build as base
from tests.context_helpers import make_context


def stage(motion, description):
    return {"id": motion.lower(), "target": "cup", "affordance": "cup rim", "motion": motion,
            "description": description, "completion": "The cup is held."}


def response(stages):
    value = {"subgoals": stages}
    return SimpleNamespace(token="", raw_text=json.dumps(value), payload={"json": value})


def override_text(monkeypatch, tmp_path, module, text):
    directory = tmp_path / module.__name__
    directory.mkdir()
    (directory / "prompt.txt").write_text(text)
    monkeypatch.setattr(module, "__file__", str(directory / "build.py"))


def test_pregrasp_merge_folds_pure_reach_into_grasp():
    client = SimpleNamespace(complete_json=lambda *a, **kw: response([
        stage("TRAVEL", "Move over the cup."), stage("GRASP", "Close on the cup."),
    ]))
    context = make_context(client=client)
    context["planner_prompt"] = planner_prompt(context)
    merged, _ = subgoal.plan(context)
    assert [s.motion for s in merged] == ["GRASP"]
    assert "Move over the cup" in merged[0].description


def test_text_retry_preserves_full_prompt(monkeypatch, tmp_path):
    calls = []

    def invalid(prompt, *args, **kwargs):
        calls.append(prompt)
        raise RuntimeError("VLM returned empty content")

    def text(prompt, *args, **kwargs):
        calls.append(prompt)
        return response([stage("GRASP", "Close on the cup.")])

    old_text = Path(subgoal.__file__).with_name("prompt.txt").read_text()
    retry_sections = old_text[old_text.index("\n[json_retry]"):]
    override_text(monkeypatch, tmp_path, subgoal, "SCENE_CONTRACT_SENTINEL\n" + retry_sections)
    override_text(monkeypatch, tmp_path, base, "TASK=${task}")
    context = make_context(client=SimpleNamespace(complete_json=invalid, complete_text=text),
                           task="move every piece", extra_context="GENERAL_CONTEXT_SENTINEL")
    context["planner_prompt"] = planner_prompt(context)
    subgoal.plan(context)
    assert len(calls) == 3
    for marker in ("GENERAL_CONTEXT_SENTINEL", "SCENE_CONTRACT_SENTINEL", "move every piece"):
        assert marker in calls[-1]
    assert context["planner_diagnostics"]["route"] == "text_retry_1"
    assert len(context["planner_diagnostics"]["errors"]) == 2


def test_empty_optional_text_keeps_prompt_unchanged(monkeypatch, tmp_path):
    override_text(monkeypatch, tmp_path, subgoal, "")
    override_text(monkeypatch, tmp_path, base, "DO ${task}")
    context = make_context(task="anything", extra_context="GENERAL")
    assert planner_prompt(context) == "DO anything\n\nGENERAL"


def test_disabled_planner_needs_no_model():
    assert subgoal.plan({"plugins": {"subgoal": False}}) == ([], "")
