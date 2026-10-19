"""Validate the explicit Python context and select LIBERO tasks."""
from __future__ import annotations

import ast
from pathlib import Path

from core.step_size import SIZE_LIMITS, size_bounds

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
NOISE_TYPES = ("none", "motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur")
FIELDS = {
    "": "sim_env robot runtime tasks libero semantic_eef_action harness model extra_context plugins plugin_state",
    "plugins": "",
    "plugin_state": "",
    "runtime": "python gl_backend seed log_dir debug",
    "tasks": "mode suite task task_ids benchmark_suites episodes initial_state",
    "libero": "original_root plus_root image_size control_hz settle_steps agentview_rotation wrist_rotation use_wrist_image perturbations",
    "libero.perturbations": "camera robot_initial_state language_variant lighting_variant table_texture_variant background_variant layout sensor_noise",
    "libero.perturbations.camera": "orbit_yaw_deg orbit_pitch_deg distance_scale yaw_deg pitch_deg",
    "libero.perturbations.layout": "distractor_variant position_level position_sample",
    "libero.perturbations.sensor_noise": "type severity",
    "semantic_eef_action": "step_m rotation_enabled rotation_step_deg sim_steps gripper_steps camera_enabled camera_step_m camera_yaw_step_deg",
    "harness": "max_steps max_subgoal_steps loop_period_s video_fps gripper_color planner_max_tokens mem_text_len coarse_step_m high_above_table_m action_chunk_steps empty_width_m open_width_m affordance_dot_radius_px",
    "model": "name api_key api_key_env base_url max_tokens temperature thinking reasoning_effort timeout_s max_retries retry_base_delay_s retry_max_delay_s",
}


def validate_context(cfg: dict) -> None:
    # Require built-in fields, while allowing users to add fields without a schema edit.
    for section, fields in FIELDS.items():
        block = cfg
        for part in section.split(".") if section else ():
            block = block.get(part, {}) if isinstance(block, dict) else {}
        if not isinstance(block, dict):
            raise ValueError(f"{section or 'context'} must be a mapping")
        missing = set(fields.split()) - set(block)
        if missing:
            raise ValueError(f"{section or 'context'}: missing {sorted(missing)}")
    if cfg["sim_env"] != "libero" or cfg["robot"] != "franka":
        raise ValueError("This launcher supports only sim_env='libero' and robot='franka'")

    def number(section, key, low, high=float("inf"), integer=False):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{key} must be a number in [{low}, {high}]")
        if integer and (not isinstance(value, int)):
            raise ValueError(f"{key} must be an integer")

    r, t, l, a, h, m = (cfg[k] for k in ("runtime", "tasks", "libero", "semantic_eef_action", "harness", "model"))
    p = l["perturbations"]
    for block, keys in ((r, ["debug"]), (l, ["use_wrist_image"]),
                        (a, ["rotation_enabled", "camera_enabled"]), (m, ["thinking"]),
                        (cfg["plugins"], list(cfg["plugins"]))):
        for key in keys:
            if not isinstance(block[key], bool):
                raise ValueError(f"{key} must be true or false")
    if r["gl_backend"] not in ("auto", "cgl", "egl", "osmesa"):
        raise ValueError("gl_backend must be auto/cgl/egl/osmesa")
    number(r, "seed", 0, 2**32 - 1, True)
    if t["mode"] not in ("single", "bundle", "benchmark") or t["suite"] not in SUITES:
        raise ValueError("Invalid task mode or suite")
    if not isinstance(t["task_ids"], list) or not isinstance(t["benchmark_suites"], list):
        raise ValueError("task_ids and benchmark_suites must be lists")
    if not t["benchmark_suites"] or any(s not in SUITES for s in t["benchmark_suites"]):
        raise ValueError(f"benchmark_suites must contain names from {SUITES}")
    number(t, "episodes", 1, integer=True)
    number(t, "initial_state", 0, integer=True)
    for key in ("image_size", "control_hz"):
        number(l, key, 1, integer=True)
    number(l, "settle_steps", 0, integer=True)
    for key in ("agentview_rotation", "wrist_rotation"):
        if l[key] not in (0, 90, 180, 270):
            raise ValueError(f"{key} must be 0, 90, 180, or 270")
    for key, hi in (("robot_initial_state", 500), ("language_variant", 50),
                    ("lighting_variant", 50), ("table_texture_variant", 28), ("background_variant", 22)):
        number(p, key, 0, hi, True)
    if p["table_texture_variant"] and p["background_variant"]:
        raise ValueError("Choose table_texture_variant OR background_variant (which includes a table)")
    for key in ("orbit_yaw_deg", "orbit_pitch_deg", "yaw_deg", "pitch_deg"):
        number(p["camera"], key, -360, 360, True)
    number(p["camera"], "distance_scale", 0.01, 10)
    scale = p["camera"]["distance_scale"] * 100
    if abs(scale - round(scale)) > 1e-7:
        raise ValueError("camera.distance_scale must use 0.01 increments")
    layout = p["layout"]
    number(layout, "distractor_variant", 0, 30, True)
    number(layout, "position_level", 0, 5, True)
    number(layout, "position_sample", 1, 4, True)
    if layout["distractor_variant"] and layout["position_level"]:
        raise ValueError("Choose distractor_variant OR position_level: upstream has separate saved states")
    if (layout["distractor_variant"] or layout["position_level"]) and (t["episodes"] != 1 or t["initial_state"] != 0):
        raise ValueError("Layout variants have one initial state: set episodes: 1 and initial_state: 0")
    noise = p["sensor_noise"]
    if noise["type"] not in NOISE_TYPES:
        raise ValueError(f"sensor_noise.type must be one of {NOISE_TYPES}")
    number(noise, "severity", 1, 10, True)
    if noise["type"] == "fog" and l["image_size"] > 256:
        raise ValueError("Upstream fog supports image_size <= 256")
    for key in ("step_m", "camera_step_m"):
        number(a, key, 0.0001, 1)
    for key in ("rotation_step_deg", "camera_yaw_step_deg"):
        number(a, key, 0.01, 180)
    for field, (low, high) in SIZE_LIMITS.items():
        # Older configurations can omit override bounds and retain the existing limits.
        bounds = dict(zip((f"min_{field}", f"max_{field}"), size_bounds(a, field)))
        number(bounds, f"min_{field}", low, high)
        number(bounds, f"max_{field}", bounds[f"min_{field}"], high)
    for key in ("sim_steps", "gripper_steps"):
        number(a, key, 1, integer=True)
    for key in ("max_steps", "max_subgoal_steps", "planner_max_tokens", "mem_text_len", "action_chunk_steps"):
        number(h, key, 1, integer=True)
    for key in ("loop_period_s", "high_above_table_m", "empty_width_m"):
        number(h, key, 0)
    number(h, "coarse_step_m", a["step_m"], 1)
    number(h, "open_width_m", h["empty_width_m"] + 0.0001, 0.1)
    number(h, "video_fps", 0.5, 30)
    number(h, "affordance_dot_radius_px", 1, 16, integer=True)
    m["name"] = str(m["name"]).lower()
    if m["name"] not in ("deepseek-flash", "kimi-k3"):
        raise ValueError("model.name must be deepseek-flash or Kimi-K3")
    effort = ("low", "high", "max")
    if m["reasoning_effort"] not in effort:
        raise ValueError(f"reasoning_effort must be one of {effort} for this model/mode")
    for key in ("max_tokens",):
        number(m, key, 1, integer=True)
    number(m, "max_retries", 0, integer=True)
    number(m, "temperature", 0, 2)
    for key in ("timeout_s", "retry_base_delay_s", "retry_max_delay_s"):
        number(m, key, 0.01)


def perturbed(cfg: dict) -> bool:
    p = cfg["libero"]["perturbations"]
    return (any(p[k] for k in ("robot_initial_state", "language_variant", "lighting_variant",
                               "table_texture_variant", "background_variant"))
            or any(v != (1.0 if k == "distance_scale" else 0) for k, v in p["camera"].items())
            or bool(p["layout"]["distractor_variant"] or p["layout"]["position_level"])
            or p["sensor_noise"]["type"] != "none")


def task_catalog(cfg: dict) -> dict:
    path = Path(cfg["libero"]["original_root"]) / "libero/libero/benchmark/libero_suite_task_map.py"
    if not path.is_file():
        raise FileNotFoundError(f"Original LIBERO task map not found: {path}; see docs/libero-launch.md")
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets))


def select_tasks(cfg: dict) -> list[tuple[str, int, str]]:
    catalog, t = task_catalog(cfg), cfg["tasks"]
    suites = t["benchmark_suites"] if t["mode"] == "benchmark" else [t["suite"]]
    selected = []
    for suite in dict.fromkeys(suites):
        names = catalog[suite]
        choices = ([t["task"]] if t["mode"] == "single" else
                   (t["task_ids"] if t["mode"] == "bundle" and t["task_ids"] else list(range(len(names)))))
        for choice in choices:
            idx = names.index(choice) if isinstance(choice, str) and choice in names else choice
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(names):
                raise ValueError(f"Unknown task {choice!r} in {suite}; use --list-tasks")
            item = (suite, idx, names[idx])
            if item not in selected:
                selected.append(item)
    return selected
