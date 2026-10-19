"""Render one before/after pair per base-frame motion token, without model calls.

Illustrations use 10 cm translations and 30-degree rotations for both EEF and camera.
These capture-only settings do not change the runtime defaults in context.py.

Run from seed_harness with the simulator Python:
../third_party/LIBERO-plus/.venv/bin/python -m scripts.illustrate_semantic_actions
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from context import create_context
from core.action_units import CAMERA_ATOMS, MOVE_ATOMS, RT_ATOMS
from core.sim.libero_config import select_tasks
from core.sim.libero_observation import observation_images
from core.sim.libero_tasks import configure_imports, create_env, task_assets
from interpreters.libero_atomic_controller import LiberoAtomicController, move_camera


TOKENS = MOVE_ATOMS + RT_ATOMS + CAMERA_ATOMS
SCENARIOS = {
    "initial_orientation": "Initial wrist orientation; default Agentview",
    "turned_and_tilted": "Wrist yaw +60°, pitch −20°; Agentview yaw +15° (preparation targets)",
}


def snapshot(env):
    """Include camera model edits and the gripper accumulator outside MuJoCo state."""
    return {
        "state": env.get_sim_state().copy(),
        "cam_pos": env.sim.model.cam_pos.copy(),
        "cam_quat": env.sim.model.cam_quat.copy(),
        "ctrl": env.sim.data.ctrl.copy(),
        "qacc_warmstart": env.sim.data.qacc_warmstart.copy(),
        "gripper_action": env.robots[0].gripper.current_action.copy(),
    }


def restore(env, saved, cfg):
    env.sim.model.cam_pos[:] = saved["cam_pos"]
    env.sim.model.cam_quat[:] = saved["cam_quat"]
    env.set_state(saved["state"])
    env.sim.data.ctrl[:] = saved["ctrl"]
    env.sim.forward()
    env.sim.data.qacc_warmstart[:] = saved["qacc_warmstart"]
    robot = env.robots[0]
    robot.gripper.current_action = saved["gripper_action"].copy()
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    obs = env.env._get_observations(force_update=True)
    return obs, LiberoAtomicController(env, cfg, obs)


def pose(env, executor):
    data = env.sim.data
    site = env.robots[0].eef_site_id
    return {
        "tcp_position_world_m": data.site_xpos[site].copy(),
        "tcp_rotation_world": data.site_xmat[site].reshape(3, 3).copy(),
        "camera_position_world_m": data.cam_xpos[executor.camera_id].copy(),
        "camera_rotation_world": data.cam_xmat[executor.camera_id].reshape(3, 3).copy(),
    }


def movement(before, after, base):
    result = {}
    for body in ("tcp", "camera"):
        result[f"{body}_translation_base_m"] = (
            base.T @ (after[f"{body}_position_world_m"] - before[f"{body}_position_world_m"])
        ).tolist()
        result[f"{body}_rotation_vector_base_deg"] = np.rad2deg(base.T @ Rotation.from_matrix(
            after[f"{body}_rotation_world"] @ before[f"{body}_rotation_world"].T
        ).as_rotvec()).tolist()
    return result


def save_views(path, obs, cfg, title):
    agent, wrist = observation_images(obs, cfg)
    width, height = agent.shape[1], agent.shape[0]
    image = Image.new("RGB", (2 * width, height + 58), "#f3f4f6")
    image.paste(Image.fromarray(agent), (0, 58))
    image.paste(Image.fromarray(wrist), (width, 58))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    draw.text((12, 7), title, fill="#111827", font=font)
    draw.text((12, 32), "Agentview", fill="#374151", font=font)
    draw.text((width + 12, 32), "Wristview", fill="#374151", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def describe(token, actions):
    directions = {"FWD": "+X", "BACK": "−X", "RIGHT": "+Y", "LEFT": "−Y",
                  "UP": "+Z", "DOWN": "−Z"}
    rotations = {"ROLL_LEFT": "+X", "ROLL_RIGHT": "−X", "PITCH_FWD": "−Y",
                 "PITCH_BACK": "+Y", "YAW_CW": "+Z", "YAW_CCW": "−Z"}
    if token in MOVE_ATOMS:
        return f"TCP {actions['step_m'] * 100:g} cm along base {directions[token[3:]]}"
    if token in RT_ATOMS:
        return f"TCP {actions['rotation_step_deg']:g}° about base {rotations[token[3:]]}"
    if token.startswith("CAM_YAW_"):
        return f"Camera {actions['camera_yaw_step_deg']:g}° about base {rotations[token[4:]]}, in place"
    return f"Camera {actions['camera_step_m'] * 100:g} cm along base {directions[token[4:]]}"


def vector(values, scale=1):
    return "[" + ", ".join(f"{value * scale:+.2f}" for value in values) + "]"


def write_gallery(output, manifest):
    eef_records = [record for record in manifest["actions"] if record["token"] not in CAMERA_ATOMS]
    position_error_mm = max(np.linalg.norm(
        np.asarray(record["commanded"]["tcp_translation_base_m"]) - record["measured"]["tcp_translation_base_m"]
    ) * 1000 for record in eef_records)
    rotation_error_deg = max(np.linalg.norm(
        np.asarray(record["commanded"]["tcp_rotation_vector_base_deg"]) - record["measured"]["tcp_rotation_vector_base_deg"]
    ) for record in eef_records)
    sections = []
    for token in TOKENS:
        rows = []
        for record in manifest["actions"]:
            if record["token"] != token:
                continue
            body = "camera" if token in CAMERA_ATOMS else "tcp"
            delta = record["measured"]
            rows.append(f'''<h3>{html.escape(SCENARIOS[record['scenario']])}</h3>
<div class="pair"><a href="{record['before_image']}"><img loading="lazy" src="{record['before_image']}" alt="{token} before"></a>
<a href="{record['after_image']}"><img loading="lazy" src="{record['after_image']}" alt="{token} after"></a></div>
<p>Measured {body}: displacement {vector(delta[body + '_translation_base_m'], 100)} cm;
rotation vector {vector(delta[body + '_rotation_vector_base_deg'])}° in base [X, Y, Z].</p>''')
        sections.append(f'<section id="{token}"><h2>{token} · {html.escape(describe(token, manifest["step_sizes"]))}</h2>'
                        + "\n".join(rows) + "</section>")
    links = " · ".join(f'<a href="#{token}">{token}</a>' for token in TOKENS)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic actions in the robot base frame</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:28px auto;padding:0 24px;max-width:1600px;color:#182230;background:#f5f7fa}}
h1{{margin-bottom:8px}}h2{{font-size:21px}}h3{{font-size:16px;font-weight:500}}
section{{background:white;padding:18px;margin:28px 0;border:1px solid #d8dee8;border-radius:10px}}
a{{color:#164fc7}}nav{{line-height:2}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
img{{width:100%;height:auto;display:block}}p{{max-width:1100px}}code{{background:#e9eef5;padding:2px 5px}}
@media(max-width:850px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>Semantic actions in the robot base frame</h1>
<p>Actual LIBERO renders, before and after <strong>one token</strong>. Each PNG contains Agentview (left)
and Wristview (right). Click an image for full size. All 20 MV / RT / CAM tokens are shown from two
starting orientations, with an identical simulator state restored before each action in a scenario.</p>
<p><strong>FWD = base +X · RIGHT = base +Y · UP = base +Z.</strong> These axes stay fixed when the wrist
or camera turns. Commanded rotations pivot at the TCP or camera; yaw CW is positive about +Z (viewed from below).
Commanded base-frame deltas match across both starting orientations. Physical results also depend on the
controller dynamics: these captures include up to {position_error_mm:.2f} mm TCP position error and
{rotation_error_deg:.2f}° rotation-vector error relative to the target after the configured control steps.
The measured values below preserve those differences.</p>
<p>The gripper is moved forward 14 cm and down 12 cm before capture for visibility. The second scenario
adds wrist yaw +60°, pitch −20°, and camera yaw +15°. Preparation uses 30 control steps per EEF action;
the illustrated actions use these capture settings: {manifest['step_sizes']['sim_steps']} control steps,
{manifest['step_sizes']['step_m'] * 100:g} cm EEF translation, {manifest['step_sizes']['rotation_step_deg']:g}° EEF rotation,
{manifest['step_sizes']['camera_step_m'] * 100:g} cm camera translation, {manifest['step_sizes']['camera_yaw_step_deg']:g}° camera yaw.
These illustration sizes leave the runtime defaults unchanged. Camera actions do not advance physics.</p>
<p>Scene: <code>{html.escape(manifest['task'])}</code>. <a href="manifest.json">Full poses, targets, measurements, and capture settings</a>.</p>
<nav>{links}</nav>{''.join(sections)}</html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "illustrate_semantic_action")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()
    cfg = create_context()
    cfg["tasks"].update(mode="single", suite=args.suite, task=args.task, initial_state=0)
    cfg["libero"].update(image_size=args.image_size, use_wrist_image=True)
    actions = cfg["semantic_eef_action"]
    actions.update(rotation_enabled=True, camera_enabled=True, step_m=0.1, rotation_step_deg=30,
                   camera_step_m=0.1, camera_yaw_step_deg=30)
    suite, _, name = select_tasks(cfg)[0]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"task": name, "suite": suite, "seed": cfg["runtime"]["seed"], "initial_state": 0,
                "step_sizes": dict(actions), "libero": cfg["libero"],
                "preparation": {"MV_FWD_m": 0.14, "MV_DOWN_m": 0.12, "sim_steps": 30,
                                "turned_and_tilted": {"RT_YAW_CW_deg": 60, "RT_PITCH_FWD_deg": 20,
                                                      "CAM_YAW_CW_deg": 15}}, "actions": []}
    with TemporaryDirectory(prefix="semantic-actions-") as temporary:
        directory = Path(temporary)
        configure_imports(cfg, directory)
        env, obs, _ = create_env(cfg, task_assets(cfg, suite, name), directory / "generated", 0)
        try:
            executor = LiberoAtomicController(env, cfg, obs)
            default_steps = actions["sim_steps"]
            actions["sim_steps"] = 30
            obs, _, _ = executor.execute("MV_FWD", obs, step_m=0.14)
            obs, _, _ = executor.execute("MV_DOWN", obs, step_m=0.12)
            scenarios = {"initial_orientation": snapshot(env)}
            obs, _, _ = executor.execute("RT_YAW_CW", obs, angle_deg=60)
            obs, _, _ = executor.execute("RT_PITCH_FWD", obs, angle_deg=20)
            move_camera(env.sim, executor.camera_id, "CAM_YAW_CW", actions["camera_step_m"],
                        15, executor.base_frame_world())
            scenarios["turned_and_tilted"] = snapshot(env)
            actions["sim_steps"] = default_steps
            for scenario, saved in scenarios.items():
                np.savez_compressed(output / f"{scenario}_state.npz", **saved)
                for token in TOKENS:
                    obs, executor = restore(env, saved, cfg)
                    before = pose(env, executor)
                    base = executor.base_frame_world()
                    before_image = f"{scenario}/{token}/before.png"
                    after_image = f"{scenario}/{token}/after.png"
                    save_views(output / before_image, obs, cfg, f"{token} | BEFORE")
                    state_before = env.get_sim_state().copy()
                    obs, commands, success = executor.execute(token, obs)
                    after = pose(env, executor)
                    target = {**before}
                    if token in CAMERA_ATOMS:
                        np.testing.assert_array_equal(env.get_sim_state(), state_before)
                        assert not commands
                        target.update(camera_position_world_m=after["camera_position_world_m"],
                                      camera_rotation_world=after["camera_rotation_world"])
                    else:
                        assert len(commands) == default_steps and not success
                        target.update(tcp_position_world_m=executor.target_position.copy(),
                                      tcp_rotation_world=executor.target_rotation.copy())
                    save_views(output / after_image, obs, cfg, f"{token} | AFTER one action")
                    manifest["actions"].append({
                        "scenario": scenario, "token": token, "base_frame_world": base.tolist(),
                        "before": {key: value.tolist() for key, value in before.items()},
                        "after": {key: value.tolist() for key, value in after.items()},
                        "commanded": movement(before, target, base),
                        "measured": movement(before, after, base), "control_steps": len(commands),
                        "before_image": before_image, "after_image": after_image,
                    })
                    print(f"Captured {scenario}/{token}", flush=True)
        finally:
            env.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_gallery(output, manifest)
    print(f"Saved {len(manifest['actions'])} pairs: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
