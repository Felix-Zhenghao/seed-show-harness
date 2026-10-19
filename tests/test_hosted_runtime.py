"""Exercise the retained model transport and roles without paid API requests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from context import create_context
from core.prompting.composer import planner_prompt
from core.prompting.content import PromptContent, PromptImage
from core.sim.libero_model import DebugTrace, LiberoVLMClient
from core.vlm.roles import ControllerAgent
from plugins.affordance import build as affordance
from plugins.decision_json import build as decision_json
from plugins.decision_output import build as decision_output
from plugins.subgoal import build as subgoal
from plugins.whole_task_stage import build as whole_task_stage

from tests.context_helpers import make_context


@pytest.fixture
def client(tmp_path):
    cfg = create_context()
    cfg["model"].update(name="deepseek-flash", api_key="test-credential", max_retries=1)
    client = LiberoVLMClient(cfg, DebugTrace(tmp_path, True))
    yield client
    client.session.close()


def reply(answer, status=200, headers=None):
    return SimpleNamespace(
        status_code=status, text=json.dumps(answer), headers=headers or {}, json=lambda: answer
    )


def completion(answer):
    return {"choices": [{"message": {"content": answer}}]}


@pytest.mark.parametrize("model", ["deepseek-flash", "kimi-k3"])
@pytest.mark.parametrize("with_wrist", [False, True])
def test_planner_controller_and_debug_transport(client, monkeypatch, model, with_wrist):
    client.model = model
    stage = dict(
        id="grasp",
        target="cup",
        affordance="cup body",
        motion="GRASP",
        description="Grasp the cup",
        completion="The cup is held",
    )
    answers = iter(
        [
            completion(json.dumps({"subgoals": [stage]})),
            completion(
                json.dumps(
                    {"decision": "MV_RIGHT", "reasoning": "WRIST: NO; PLAN: MV_RIGHT, MV_UP"}
                )
            ),
        ]
    )
    requests = []

    def post(url, json, timeout):
        requests.append(json)
        return reply(next(answers))

    monkeypatch.setattr(client.session, "post", post)
    scene = np.full((8, 8, 3), 42, dtype=np.uint8)
    wrist = np.full_like(scene, 84) if with_wrist else None
    context = make_context(client=client, task="Move the cup", agentview_image=scene,
                           wrist_image=wrist, extra_context="Keep the cup upright.")
    context["harness"]["planner_max_tokens"] = 4567
    context["debug"] = True
    context["planner_prompt"] = planner_prompt(context)
    subgoals, _ = subgoal.plan(context)
    assert subgoals[0].to_prompt_dict() == stage
    cfg = create_context()
    cfg["extra_context"] = "world +Y is right"
    agent = ControllerAgent(client)
    context = make_context(
        cfg, client=client, task="Move the cup", subgoal=stage, recent_moves=["MV_UP"],
        previous_direction="MV_UP", agentview_image=scene, wrist_image=wrist,
    )
    decision = agent.decide(context, debug=True)
    assert decision.token == "MV_RIGHT"
    assert decision.payload["target_in_wrist"] is False
    assert decision.payload["chunk_plan"] == ["MV_RIGHT", "MV_UP"]
    assert "10.0 cm above the table" in agent.last_prompt
    budget_key = "max_completion_tokens" if model == "kimi-k3" else "max_tokens"
    assert requests[0][budget_key] == 4567
    assert requests[1][budget_key] == client.max_tokens
    for payload in requests:
        blocks = payload["messages"][0]["content"]
        assert [b["type"] for b in blocks] == ["text"] + ["image_url"] * (2 if with_wrist else 1)
        assert all(b["image_url"]["url"].startswith("data:image/png;base64,") for b in blocks[1:])
        assert (
            "response_format" in payload
            if model == "deepseek-flash"
            else "response_format" not in payload
        )
    trace = client.trace.directory
    assert len(list((trace / "calls").glob("*_image_*.png"))) == (4 if with_wrist else 2)
    assert len(list((trace / "calls").glob("*_response.json"))) == 2
    assert "test-credential" not in (trace / "trajectory.html").read_text()


def test_controller_parse_retry_uses_enabled_actions(client, monkeypatch):
    answers = iter([completion("unparseable"), completion("MV_UP")])
    payloads = []

    def post(url, json, timeout):
        payloads.append(json)
        return reply(next(answers))

    monkeypatch.setattr(client.session, "post", post)
    cfg = create_context()
    agent = ControllerAgent(client)
    result = agent.decide(make_context(cfg, client=client))
    assert result.token == "MV_UP" and not result.payload.get("fallback")
    assert len(payloads) == 2
    assert "response_format" not in payloads[1]
    assert payloads[1]["max_tokens"] == client.max_tokens


def test_controller_output_rules_survive_in_prefix_when_observations_change(client, monkeypatch):
    payloads = []

    def post(url, json, timeout):
        payloads.append(json)
        return reply(completion('{"decision":"MV_UP","reasoning":"Lift clear."}'))

    monkeypatch.setattr(client.session, "post", post)
    context = make_context(
        client=client, task="Move the cup", recent_moves=["MV_RIGHT"],
        subgoal={"motion": "GRASP", "target": "cup", "affordance": "rim"},
    )
    context["semantic_eef_action"].update(max_distance_m=0.05, max_angle_deg=30)
    contracts = [decision_output.build(context), decision_json.build(context)]
    agent = ControllerAgent(client)
    agent.decide(context)
    context.update(task="Open the drawer", subgoal={"motion": "PULL", "affordance": "handle"},
                   gripper_state="CLOSED", recent_moves=["GRASP"], recovery="auto_release",
                   agentview_image=np.ones((8, 8, 3), dtype=np.uint8), wrist_image=None)
    context["proprioception"]["eef_pos"][2] = 1.0
    agent.decide(context)

    before, after = [p["messages"][0]["content"] for p in payloads]
    first_text, second_text = before[0]["text"], after[0]["text"]
    prefix = first_text.split("Your current task:\n", 1)[0]
    assert second_text.split("Your current task:\n", 1)[0] == prefix
    for contract in contracts:
        assert contract in prefix
        assert first_text.count(contract) == second_text.count(contract) == 1
    assert "[0.0001, 0.05]" in prefix and "[0.01, 30]" in prefix
    assert "Open the drawer" in second_text and '"affordance": "handle"' in second_text
    assert '"gripper_state": "CLOSED"' in second_text and "20.0 cm above the table" in second_text
    assert "Recent moves, newest first: GRASP" in second_text
    assert "automatically reopened" in second_text and "Move the cup" not in second_text
    assert [b["type"] for b in before] == ["text", "image_url", "image_url"]
    assert [b["type"] for b in after] == ["text", "image_url"]
    assert before[1]["image_url"] != after[1]["image_url"]


def test_controller_repairs_reuse_live_plugin_results_and_images_once(client, monkeypatch):
    builds, payloads = [], []
    marker = np.full((2, 2, 3), 100, dtype=np.uint8)

    def mark(context):
        builds.append(context)
        context["agentview_image"] = np.full((8, 8, 3), 42, dtype=np.uint8)
        context["wrist_image"] = np.full((8, 8, 3), 84, dtype=np.uint8)
        return PromptContent(("Current contact:", PromptImage(marker, "high"), "Keep this contact."))

    monkeypatch.setattr(affordance, "build", mark)
    answers = iter(["invalid", "invalid", "MV_UP"])

    def post(url, json, timeout):
        payloads.append(json)
        return reply(completion(next(answers)))

    monkeypatch.setattr(client.session, "post", post)
    context = make_context(client=client, task="Move ${literal} cup", recent_moves=["MV_RIGHT"])
    context["plugins"]["affordance"] = True
    # Plugin text resembling a section heading must not become a parsing boundary.
    context["extra_context"] = "Literal heading: Your current task:\nKeep this exact instruction."
    agent = ControllerAgent(client)
    assert agent.decide(context).token == "MV_UP"
    assert len(builds) == 1 and len(payloads) == 3

    first, retry, strict = [p["messages"][0]["content"] for p in payloads]
    contracts = [decision_output.build(context), decision_json.build(context)]
    for blocks in (first, retry, strict):
        assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url", "image_url"]
        assert context["extra_context"] in blocks[0]["text"]
        assert "Move ${literal} cup" in blocks[0]["text"]
        assert blocks[1] == first[1]  # The plugin's inline image and detail survive.
        assert blocks[3:] == first[3:]  # The same marked Agentview/Wristview reach every retry.
        assert blocks[2]["text"].startswith("Keep this contact.")
        assert "Recent moves, newest first: MV_RIGHT" in blocks[2]["text"]
    for contract in contracts:
        assert first[0]["text"].index(contract) < first[0]["text"].index("Move ${literal} cup")
        for blocks in (retry, strict):
            assert all(contract not in b.get("text", "") for b in blocks)
            assert '"distance_m"' not in "\n".join(b.get("text", "") for b in blocks)
    assert '"step_size_reasoning"' in agent.last_prompt


def test_configured_plugins_reach_controller_and_transport_repairs(client, monkeypatch, tmp_path):
    import importlib
    from pathlib import Path

    for name in ("decision_output", "decision_token", "strict_token_retry"):
        module = importlib.import_module(f"plugins.{name}.build")
        directory = tmp_path / name
        directory.mkdir()
        text = Path(module.__file__).with_name("prompt.txt").read_text()
        (directory / "prompt.txt").write_text(text + f"\nCUSTOM_{name}\n")
        monkeypatch.setattr(module, "__file__", str(directory / "build.py"))
    cfg = create_context()
    answers = iter([completion("invalid"), completion("invalid"), completion("MV_UP")])
    payloads = []

    def post(url, json, timeout):
        payloads.append(json)
        return reply(next(answers))

    monkeypatch.setattr(client.session, "post", post)
    agent = ControllerAgent(client)
    response = agent.decide(make_context(cfg, client=client, agentview_image=None, wrist_image=None))
    assert response.token == "MV_UP"
    prompts = [payload["messages"][0]["content"][0]["text"] for payload in payloads]
    assert "CUSTOM_decision_output" in prompts[0]
    assert "CUSTOM_decision_token" in prompts[1]
    assert "CUSTOM_strict_token_retry" in prompts[2]


def test_transient_error_retries_but_auth_error_stops(client, monkeypatch):
    calls, delays = [], []
    responses = iter([reply({}, 429, {"Retry-After": "3"}), reply(completion("MV_UP"))])

    def post(url, json, timeout):
        calls.append(json)
        return next(responses)

    monkeypatch.setattr(client.session, "post", post)
    monkeypatch.setattr("core.vlm.vlm_client.time.sleep", delays.append)
    monkeypatch.setattr("core.vlm.vlm_client.random.uniform", lambda *args: 1)
    assert client.complete_token("Choose", ["MV_UP"], None).token == "MV_UP"
    assert len(calls) == 2 and delays == [4]
    monkeypatch.setattr(
        client.session, "post", lambda *args, **kwargs: reply({"error": "test-credential"}, 401)
    )
    with pytest.raises(ConnectionError, match="REDACTED"):
        client.complete_token("Choose", ["MV_UP"], None)
    assert delays == [4]


def test_custom_context_reaches_roles_images_and_every_retry(client, monkeypatch):
    from plugins.observations import build as observations

    seen = []
    original_build = observations.build

    def inspect_images(context):
        assert context["object_notes"] == "user supplied scene data"
        seen.append(context["object_notes"])
        return original_build(context)

    monkeypatch.setattr(observations, "build", inspect_images)
    context = make_context(client=client, object_notes="user supplied scene data")
    answers = iter(["invalid", "invalid", "MV_UP"] + ["invalid"] * 4)
    monkeypatch.setattr(client.session, "post", lambda *a, **kw: reply(completion(next(answers))))
    assert ControllerAgent(client).decide(context).token == "MV_UP"
    context["planner_prompt"] = planner_prompt(context)
    context["fallback_stage"] = json.loads(whole_task_stage.build({**context, "fallback": True}))
    subgoal.plan(context)
    assert context["planner_response"].payload["planner_fallback"]
    assert len(seen) == 7


def test_local_credentials_keep_shell_precedence(tmp_path, monkeypatch):
    from core import secrets

    path = tmp_path / "secrets.env"
    path.write_text("DEEPSEEK_API_KEY=base\nMOONSHOT_API_KEY=base\n")
    path.with_name("secrets.local.env").write_text(
        'export DEEPSEEK_API_KEY="local"\nMOONSHOT_API_KEY=local\n'
    )
    monkeypatch.setattr(secrets, "DEFAULT_SECRETS_ENV", path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "exported")
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    secrets.load_secrets_env()
    import os

    assert os.environ["DEEPSEEK_API_KEY"] == "exported"
    assert os.environ["MOONSHOT_API_KEY"] == "local"
