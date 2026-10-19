"""Resolve named perturbations into upstream assets without modifying either checkout."""
from __future__ import annotations

import ast
import copy
import inspect
import os
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from core.sim.libero_config import perturbed


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required LIBERO asset is missing: {path}. "
                                "Install the matching upstream assets or choose an available variant.")
    return path


def task_assets(cfg: dict, suite: str, name: str) -> dict:
    """Resolve original and variant assets independently, including layout-specific states."""
    original = Path(cfg["libero"]["original_root"]) / "libero/libero"
    plus = Path(cfg["libero"]["plus_root"]) / "libero/libero"
    p = cfg["libero"]["perturbations"]
    base = require_file(original / "bddl_files" / suite / f"{name}.bddl")
    paths = {"base": base, "layout": base}

    def variant(suffix):
        return require_file(plus / "bddl_files" / suite / f"{name}{suffix}.bddl")

    layout = p["layout"]
    suffix = ""
    if layout["distractor_variant"]:
        suffix = f"_add_{layout['distractor_variant']}"
    elif layout["position_level"]:
        suffix = f"_level{layout['position_level']}_sample{layout['position_sample']}"
    if suffix:
        # Goal-suite displacement assets use an extra upstream "_moved" marker.
        # It is an asset naming detail, not a different user-facing task name.
        layout_name = name
        if layout["position_level"] and not (plus / "bddl_files" / suite / f"{name}{suffix}.bddl").is_file():
            layout_name += "_moved"
        paths["layout"] = require_file(plus / "bddl_files" / suite / f"{layout_name}{suffix}.bddl")
        paths["initial_states"] = require_file(
            plus / "init_files/libero_newobj" / suite / f"{layout_name}{suffix}.pruned_init")
    else:
        paths["initial_states"] = require_file(original / "init_files" / suite / f"{name}.pruned_init")
    for key, marker in (("language", "language_variant"), ("light", "lighting_variant"),
                        ("table", "table_texture_variant"), ("tb", "background_variant")):
        if p[marker]:
            paths[key] = variant(f"_{key}_{p[marker]}")
    return paths


def problem_name(text: str) -> str:
    match = re.search(r"\(problem\s+([^\s()]+)\)", text, re.I)
    if not match:
        raise ValueError("BDDL is missing a problem declaration")
    return match.group(1)


def language(text: str) -> str:
    match = re.search(r"\(:language\s+([^()]*)\)", text, re.I)
    if not match:
        raise ValueError("BDDL is missing its language instruction")
    return " ".join(match.group(1).split())


def _scene_settings(problem: str, root: Path) -> tuple[Path, dict]:
    from libero.libero.envs import TASK_MAPPING

    cls = TASK_MAPPING[problem.lower()]
    # Upstream scene subclasses declare literal kwargs; read those defaults without
    # instantiating a second physics environment or copying the long class registry.
    scene, properties = None, None
    for parent in cls.__mro__:
        if parent.__module__.startswith("libero."):
            tree = ast.parse(textwrap.dedent(inspect.getsource(parent)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in ("scene_xml", "scene_properties"):
                        try:
                            val = ast.literal_eval(value)
                        except (ValueError, TypeError):
                            continue
                        if key.value == "scene_xml" and scene is None:
                            scene = val
                        elif key.value == "scene_properties" and properties is None:
                            properties = val
            if scene is not None and properties is not None:
                break
    if scene is None:
        raise ValueError(f"Cannot resolve upstream scene for {problem}")
    return require_file(root / "assets" / scene), properties or {}


def _combined_scene(background: Path, lights: Path, output: Path) -> None:
    scene, illumination = ET.parse(background), ET.parse(lights)
    # Resolve file paths before relocating the XML into the run directory.
    for element in scene.iter():
        if element.get("file"):
            element.set("file", str((background.parent / element.get("file")).resolve()))
        for child in list(element):
            if child.tag == "light":
                element.remove(child)
    world = scene.getroot().find("worldbody")
    source = illumination.getroot().find("worldbody")
    if world is None or source is None:
        raise ValueError("Upstream scene XML is missing worldbody")
    if len(source.findall(".//light")) != len(source.findall("light")):
        raise ValueError("Cannot combine nested upstream lights with this background")
    for light in source.findall("light"):
        world.append(copy.deepcopy(light))
    for tag in ("visual", "option"):
        replacement = illumination.getroot().find(tag)
        if replacement is not None:
            old = scene.getroot().find(tag)
            if old is not None:
                scene.getroot().remove(old)
            scene.getroot().append(copy.deepcopy(replacement))
    scene.write(output, encoding="unicode")


def prepare_task(cfg: dict, assets: dict, directory: Path) -> tuple[str, str, dict]:
    """Compose layout, language, and scene; encode only the wrapper's virtual suffix."""
    if not perturbed(cfg):
        text = assets["base"].read_text()
        return str(assets["base"]), language(text), {}
    directory.mkdir(parents=True, exist_ok=True)
    text = assets["layout"].read_text()
    # Preserve the original instruction unless a paraphrase was explicitly requested.
    instruction = language(assets.get("language", assets["base"]).read_text())
    text = re.sub(r"\(:language\s+[^()]*\)", lambda _: f"(:language {instruction})", text, flags=re.I)
    background = assets.get("tb", assets.get("table"))
    scene = background or assets.get("light")
    kwargs = {}
    if scene:
        problem = problem_name(scene.read_text())
        text = re.sub(r"\(problem\s+[^()]+\)", lambda _: f"(problem {problem})", text, count=1, flags=re.I)
    if background and "light" in assets:
        root = Path(cfg["libero"]["plus_root"]) / "libero/libero"
        background_xml, properties = _scene_settings(problem_name(background.read_text()), root)
        light_xml, _ = _scene_settings(problem_name(assets["light"].read_text()), root)
        combined = directory / "scene.xml"
        _combined_scene(background_xml, light_xml, combined)
        kwargs = {"scene_xml": str(combined), "scene_properties": properties}
    bddl = directory / "task.bddl"
    bddl.write_text(text)
    p, c = cfg["libero"]["perturbations"], cfg["libero"]["perturbations"]["camera"]
    # LIBERO-Plus strips this virtual suffix before opening task.bddl. No user ever
    # has to invent a filename, and no upstream benchmark files are renamed.
    virtual = (f"{bddl.with_suffix('')}_view_{c['orbit_yaw_deg']}_{c['orbit_pitch_deg']}_"
               f"{round(c['distance_scale'] * 100)}_{c['yaw_deg']}_{c['pitch_deg']}_"
               f"initstate_{p['robot_initial_state']}.bddl")
    return virtual, instruction, kwargs


def configure_imports(cfg: dict, run_dir: Path) -> None:
    """Select original or Plus before importing libero; never touch ~/.libero."""
    key = "plus_root" if perturbed(cfg) else "original_root"
    root = Path(cfg["libero"][key])
    require_file(root / "libero/libero/__init__.py")
    if "libero.libero" in sys.modules:
        raise RuntimeError("LIBERO already imported; launch each configuration in a fresh process")
    sys.path.insert(0, str(root))
    config_dir = run_dir / "libero_paths"
    config_dir.mkdir(parents=True)
    package = root / "libero/libero"
    dataset_dir = config_dir / "datasets"
    dataset_dir.mkdir()
    paths = {"benchmark_root": str(package), "assets": str(package / "assets"),
             "bddl_files": str(package / "bddl_files"), "init_states": str(package / "init_files"),
             "datasets": str(dataset_dir)}
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    backend = cfg["runtime"]["gl_backend"]
    os.environ["MUJOCO_GL"] = ("cgl" if sys.platform == "darwin" else "egl") if backend == "auto" else backend


def create_env(cfg: dict, assets: dict, directory: Path, episode: int):
    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    bddl, instruction, kwargs = prepare_task(cfg, assets, directory)
    l, a, h = cfg["libero"], cfg["semantic_eef_action"], cfg["harness"]
    states = np.asarray(torch.load(assets["initial_states"], weights_only=False))
    if states.ndim == 1:
        states = states[None, :]
    index = cfg["tasks"]["initial_state"] + episode
    if index >= len(states):
        raise ValueError(f"initial_state {index} exceeds the {len(states)} states in {assets['initial_states']}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=l["image_size"], camera_widths=l["image_size"],
        control_freq=l["control_hz"], hard_reset=False, ignore_done=True,
        horizon=(h["max_steps"] + 1) * max(a["sim_steps"], a["gripper_steps"]) + l["settle_steps"],
        **kwargs,
    )
    try:
        env.seed(cfg["runtime"]["seed"] + episode)
        # Bypass upstream's unbounded reset retry loop; a broken placement must fail.
        env.env.reset()
        env.set_init_state(states[index])
        if l["perturbations"]["robot_initial_state"]:
            robot = env.robots[0]
            env.sim.data.qpos[robot._ref_joint_pos_indexes] = robot.robot_model.init_qpos
            env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
            env.sim.forward()
        env.robots[0].controller.update(force=True)
        env.robots[0].controller.reset_goal()
        # Restore first, then settle. This also ensures the first VLM sees a real render.
        for _ in range(l["settle_steps"]):
            env.step([0, 0, 0, 0, 0, 0, -1])
        obs = env.env._get_observations(force_update=True)
        return env, obs, instruction
    except BaseException:
        env.close()
        raise
