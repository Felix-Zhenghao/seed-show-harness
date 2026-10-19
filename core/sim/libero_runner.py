"""Launch LIBERO with the editable object in context.py."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from context import create_context, observe_context, read_proprioception
from core.prompting.composer import planner_prompt
from plugins.action_chunk import build as action_chunk
from plugins.auto_release import build as auto_release
from plugins.mem_text import build as mem_text
from plugins.recovery import build as recovery
from plugins.subgoal import build as subgoal
from plugins.variable_step import build as variable_step
from plugins.whole_task_stage import build as whole_task_stage
from core.record.episode_logger import EpisodeLogger
from core.record.images import save_png
from core.step_size import resolve_step_size
from core.sim.libero_config import perturbed, select_tasks, task_catalog, validate_context
from core.sim.libero_observation import observation_images
from core.sim.libero_model import DebugTrace, LiberoVLMClient, write_json
from core.sim.libero_tasks import configure_imports, create_env, task_assets
from core.types import EpisodeResult, Subgoal
from core.vlm.roles import ControllerAgent
from interpreters.libero_atomic_controller import LiberoAtomicController


def state_record(env, obs: dict) -> dict:
    camera = env.sim.model.camera_name2id("agentview")
    return {**read_proprioception(obs), "camera_world_position": env.sim.data.cam_xpos[camera].tolist(),
            "tcp_rotation_world": env.sim.data.site_xmat[env.robots[0].eef_site_id].reshape(3, 3).tolist(),
            "camera_world_rotation": env.sim.data.cam_xmat[camera].reshape(3, 3).tolist()}


def measured_movement(before: dict, after: dict) -> dict:
    """Measure achieved TCP motion, including contact/settling effects."""
    translation = np.asarray(after["eef_pos"]) - np.asarray(before["eef_pos"])
    rotation = (np.asarray(after["tcp_rotation_world"]) @
                np.asarray(before["tcp_rotation_world"]).T)
    return {
        "translation_world_m": translation.tolist(),
        "distance_m": float(np.linalg.norm(translation)),
        "angle_deg": float(np.rad2deg(Rotation.from_matrix(rotation).magnitude())),
    }


def run_episode(cfg: dict, task: tuple, episode: int, root: Path) -> EpisodeResult:
    suite, task_id, name = task
    h = cfg["harness"]
    episode_context = {**cfg, "plugin_state": {}}
    logger = EpisodeLogger(root / suite / f"episode_{episode}", task_id, video_fps=h["video_fps"])
    trace = DebugTrace(logger.run_dir, cfg["runtime"]["debug"])
    env = client = None
    success, reason, steps = False, "max_steps_exceeded", 0
    try:
        assets = task_assets(cfg, suite, name)
        logger.write_metadata({"config": cfg, "suite": suite, "task_id": task_id, "name": name,
                               "episode": episode, "assets": {k: str(v) for k, v in assets.items()}})
        client = LiberoVLMClient(cfg, trace)
        env, obs, task_text = create_env(cfg, assets, logger.run_dir / "generated", episode)
        executor = LiberoAtomicController(env, cfg, obs)
        agent = ControllerAgent(client)
        initial_context = observe_context(
            episode_context, executor=executor, observation=obs, client=client, task=task_text,
            trace=trace, debug=trace.enabled,
        )
        agentview, wrist = initial_context["agentview_image"], initial_context["wrist_image"]
        trace.event("Initial observation", {"task": task_text, "state": state_record(env, obs)})
        save_png(logger.run_dir / "initial_agentview.png", agentview)
        if wrist is not None:
            save_png(logger.run_dir / "initial_wrist.png", wrist)
        success = bool(env.check_success())
        if success:
            reason = "simulator_success_at_reset"
        if cfg["plugins"].get("subgoal") and not success:
            trace.context = {"role": "planner", "step": 0, "task": task_text}
            initial_context["planner_prompt"] = planner_prompt(initial_context)
            fallback = whole_task_stage.build({**initial_context, "fallback": True})
            initial_context["fallback_stage"] = json.loads(fallback) if fallback else None
            subgoals, raw = subgoal.plan(initial_context)
            logger.save_planner_prompt(1, initial_context["planner_prompt_text"])
            logger.write_planner_diagnostics(initial_context["planner_diagnostics"])
        else:
            stage_text = whole_task_stage.build(initial_context)
            if not stage_text:
                raise ValueError("Enable subgoal or whole_task_stage in context['plugins']")
            subgoals = [Subgoal.from_dict(json.loads(stage_text), index=0)]
            raw = "Single whole-task stage"
        logger.write_plan({"task": task_text, "raw_plan": raw,
                           "subgoals": [s.to_prompt_dict() for s in subgoals]})
        stage, stage_steps, recent, notice = 0, 0, [], None
        target_in_wrist = None
        for index in range(h["max_steps"]):
            if success:
                break
            started = time.monotonic()
            if stage_steps >= h["max_subgoal_steps"]:
                reason = "subgoal_step_cap_exceeded"
                break
            step_state = state_record(env, obs)
            stage_before = stage
            trace.context = {"role": "controller", "step": index, "stage": stage,
                             "subgoal": subgoals[stage].to_prompt_dict(), "state": step_state}
            if trace.enabled:
                np.savez_compressed(logger.run_dir / f"state_{index:04d}.npz", state=env.get_sim_state(),
                                    camera_pos=env.sim.model.cam_pos.copy(), camera_quat=env.sim.model.cam_quat.copy())
            context = observe_context(
                episode_context, executor=executor, observation=obs, client=client, task=task_text,
                subgoal=subgoals[stage].to_prompt_dict(), stage_key=str(stage),
                stage_index=stage, subgoals=subgoals, recent_moves=recent, recovery=notice,
                images=(agentview, wrist), trace=trace, debug=trace.enabled,
                target_in_wrist=target_in_wrist,
            )
            intervention = recovery.before_decision(context)
            queued_token = (None if intervention and intervention.token
                            else action_chunk.next_action(context))
            source = "controller"
            response = None  # Sizes and reasoning belong only to this step's decision.
            affordance_record = None
            if intervention and intervention.token:
                token, source, response = intervention.token, "recovery", None
                notice = intervention
                action_chunk.clear(context)
                if intervention.rollback_index is not None:
                    stage, stage_steps = intervention.rollback_index, 0
                if intervention.reset_history:
                    recent.clear()
            elif queued_token is not None:
                token, source = queued_token, "action_chunk"
            else:
                response = agent.decide(context, debug=trace.enabled)
                agentview, wrist = context["agentview_image"], context["wrist_image"]
                affordance_record = context["affordance"]
                # A parse failure is visible as a failure, never an invented move.
                if response.payload.get("fallback"):
                    raise ValueError("Controller returned no valid action after retry; see model trace")
                token = response.token
                target_in_wrist = response.payload.get("target_in_wrist")
                context["token"] = token
                action_chunk.queue_plan(context)
                if trace.enabled:
                    logger.save_controller_prompt(index, agent.last_prompt)
            context["token"] = token
            context["stage_index"] = stage
            context["subgoal"] = subgoals[stage].to_prompt_dict()
            size = resolve_step_size(
                token, response.payload.get("step_size") if response is not None else None,
                distance_m=variable_step.step_m(context),
                angle_deg=cfg["semantic_eef_action"]["rotation_step_deg"],
            )
            new_obs, commands, success = executor.execute(
                token, obs, step_m=size["effective_distance_m"], angle_deg=size["effective_angle_deg"],
            )
            # Capture this semantic action before any automatic RELEASE adds settling motion.
            size["measured"] = measured_movement(step_state, state_record(env, new_obs))
            done = token == "DONE"
            context.update(
                observation=new_obs, proprioception=read_proprioception(new_obs), success=success,
                gripper_state="CLOSED" if executor.gripper > 0 else "OPEN", subgoal_done=done,
            )
            after = recovery.after_step(context)
            if after and not success:
                notice = after
                done = done and not after.block_done
                if after.rollback_index is not None:
                    stage, stage_steps = after.rollback_index, 0
                if after.reset_history:
                    recent.clear()
                    action_chunk.clear(context)
            new_obs, release_commands, success, auto_released = auto_release.after_step(
                context, requested=bool(after and after.release))
            commands.extend(release_commands)
            if auto_released:
                action_chunk.clear(context)
                done = False
                notice = "auto_release"
                print(f"[{suite}/{task_id} episode {episode} step {index}] Automatic RELEASE", flush=True)
            steps, stage_steps = index + 1, stage_steps + 1
            context["recent_moves"] = recent
            recent = mem_text.remember(context, [token] + (["RELEASE"] if auto_released else []))
            output = response.payload.get("json", {}) if response is not None else {}
            reason_note = notice.reason if hasattr(notice, "reason") else (notice or "")
            record = {"i": index, "task": task_text, "stage": subgoals[stage_before].to_prompt_dict(),
                      "stage_index": stage_before, "action": token, "action_source": source,
                      "vlm": {"c": {"decision": token, "why": output.get("reasoning", reason_note),
                                    "step_size_reasoning": output.get("step_size_reasoning"),
                                    "raw": response.raw_text if response is not None else ""}},
                      "step_size": size,
                      "before": step_state, "after": state_record(env, new_obs),
                      "control_actions": commands, "success": success,
                      "auto_release": auto_released, "affordance": affordance_record,
                      "executed_tokens": [token] + (["RELEASE"] if auto_released else []),
                      "recovery": asdict(after or intervention) if (after or intervention) else None}
            logger.log_step(index, agentview, wrist, record)
            trace.event(f"Action {index}: {token}", record, [f"images/agentview/{index:04d}.png"])
            print(f"[{suite}/{task_id} episode {episode} step {index}] {token}: {output.get('reasoning', reason_note)}", flush=True)
            obs = new_obs
            agentview, wrist = observation_images(obs, cfg)
            if success:
                reason = "simulator_success"
            elif done:
                stage += 1
                stage_steps, recent, notice = 0, [], None
                action_chunk.clear(context)
                if stage >= len(subgoals):
                    reason = "agent_done_without_simulator_success"
                    break
            delay = h["loop_period_s"] - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
        save_png(logger.run_dir / "final_agentview.png", agentview)
        if wrist is not None:
            save_png(logger.run_dir / "final_wrist.png", wrist)
        if trace.enabled:
            np.savez_compressed(logger.run_dir / "state_final.npz", state=env.get_sim_state(),
                                camera_pos=env.sim.model.cam_pos.copy(), camera_quat=env.sim.model.cam_quat.copy())
            trace.event("Final observation", {"state": state_record(env, obs)}, ["final_agentview.png"])
    except BaseException as exc:
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else f"error: {exc}"
        trace.event("Episode stopped", {"reason": reason})
        raise
    finally:
        summary = {"success": success, "steps": steps, "end_reason": reason, "model_calls": trace.calls}
        logger.write_summary(summary)
        trace.event("Summary", summary)
        try:
            video = logger.close(success, h["video_fps"])
        finally:
            if client is not None:
                client.session.close()
            if env is not None:
                env.close()
    return EpisodeResult(success, steps, reason, str(video), str(logger.run_dir))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate configuration and selected task assets without model calls")
    parser.add_argument("--list-tasks", action="store_true", help="List original task indices and names without running")
    args = parser.parse_args(argv)
    cfg = create_context()
    validate_context(cfg)
    if args.list_tasks:
        for suite, names in task_catalog(cfg).items():
            for i, name in enumerate(names):
                print(f"{suite} {i}: {name}")
        return 0
    selected = select_tasks(cfg)
    for suite, _, name in selected:
        task_assets(cfg, suite, name)
    print(f"{'LIBERO-Plus' if perturbed(cfg) else 'Original LIBERO'}: {len(selected)} task(s), "
          f"{cfg['tasks']['episodes']} episode(s) each; model={cfg['model']['name']}", flush=True)
    if args.check:
        for suite, idx, name in selected:
            print(f"  {suite}/{idx}: {name}")
        return 0
    root = Path(cfg["runtime"]["log_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    write_json(root / "context.resolved.json", cfg)
    print(f"Logs: {root}", flush=True)
    configure_imports(cfg, root)
    results = []
    try:
        for task in selected:
            for episode in range(cfg["tasks"]["episodes"]):
                random.seed(cfg["runtime"]["seed"] + episode)
                np.random.seed(cfg["runtime"]["seed"] + episode)
                result = run_episode(cfg, task, episode, root)
                results.append({"suite": task[0], "task_id": task[1], "task_name": task[2],
                                "episode": episode, **asdict(result)})
                write_json(root / "results.json", results)
    except (Exception, KeyboardInterrupt) as exc:
        write_json(root / "error.json", {"error": str(exc), "completed_episodes": len(results)})
        print(f"Run stopped: {exc}\nLogs: {root}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(f"Completed: {sum(r['success'] for r in results)}/{len(results)} successful; results: {root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
