"""Edit constants in create_context(); edit live values in observe_context()."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("SHOW_HARNESS_ROOT", Path(__file__).resolve().parent.parent)).resolve()


def create_context() -> dict:
    """A fresh, editable context for each run. No simulator or model calls here."""
    return {
        "sim_env": "libero",
        "robot": "franka",
        "runtime": {
            "python": str(ROOT / "third_party/LIBERO-plus/.venv/bin/python"),
            "gl_backend": "auto",  # cgl on macOS, egl on Linux; also accepts osmesa.
            "seed": 0,
            "log_dir": str(ROOT / "runs/libero"),
            "debug": True,  # Save exact model inputs, replies, and simulator states.
        },
        "tasks": {
            "mode": "single",  # single, bundle, or benchmark.
            "suite": "libero_10",
            "task": 9,  # Original suite index or full name; see ./run.sh --list-tasks.
            "task_ids": [],  # bundle: an empty list selects the whole suite.
            "benchmark_suites": [
                "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90",
            ],
            "episodes": 1,
            "initial_state": 0,  # Consecutive episodes use consecutive saved states.
        },
        "libero": {
            "original_root": str(ROOT / "third_party/LIBERO"),
            "plus_root": str(ROOT / "third_party/LIBERO-plus"),
            "image_size": 256,
            "control_hz": 20,
            "settle_steps": 10,
            "agentview_rotation": 180,  # Counterclockwise degrees: 0/90/180/270.
            "wrist_rotation": 180,  # The training dataset uses 0; VLM input uses 180.
            "use_wrist_image": True,
            "perturbations": {
                "camera": {
                    "orbit_yaw_deg": 0,
                    "orbit_pitch_deg": 0,
                    "distance_scale": 1.0,  # Multiples of 0.01; >1 moves farther away.
                    "yaw_deg": 0,
                    "pitch_deg": 0,
                },
                "robot_initial_state": 0,  # 0 original; 1..500 upstream Panda poses.
                "language_variant": 0,  # 0 original; 1..50 paraphrases.
                "lighting_variant": 0,  # 0 original; 1..50 lighting presets.
                "table_texture_variant": 0,  # 0 original; 1..28 textures.
                "background_variant": 0,  # 1..22; exclusive with table_texture_variant.
                "layout": {
                    "distractor_variant": 0,  # 1..30; exclusive with position_level.
                    "position_level": 0,  # 0 original; 1..5 displacement severities.
                    "position_sample": 1,  # 1..4; layout variants have one saved state.
                },
                "sensor_noise": {
                    # none, motion_blur, gaussian_blur, zoom_blur, fog, glass_blur.
                    "type": "none",
                    "severity": 1,  # 1..10; external camera only.
                },
            },
        },
        "semantic_eef_action": {
            "step_m": 0.02,  # Robot base axes: FWD/RIGHT/UP = base +X/+Y/+Z.
            "rotation_enabled": True,
            "rotation_step_deg": 15,  # Base-fixed rotation axes through the TCP.
            # Inclusive bounds for model overrides only; omitted/null sizes use the defaults.
            "min_distance_m": 0.0001,
            "max_distance_m": 1.0,
            "min_angle_deg": 0.01,
            "max_angle_deg": 180.0,
            "sim_steps": 10,  # Hold each Cartesian target for this many control steps.
            "gripper_steps": 15,
            "camera_enabled": True,
            "camera_step_m": 0.1,  # Along the same robot base axes as MV_*.
            "camera_yaw_step_deg": 10,
        },
        "harness": {
            "max_steps": 20,
            "max_subgoal_steps": 45,
            "loop_period_s": 0.0,
            "video_fps": 2,
            "gripper_color": "black",
            "planner_max_tokens": 24000,
            "affordance_dot_radius_px": 3,
            "mem_text_len": 10,
            "coarse_step_m": 0.04,
            "high_above_table_m": 0.08,
            "action_chunk_steps": 3,
            "empty_width_m": 0.005,
            "open_width_m": 0.07,
        },
        "model": {
            "name": "deepseek-flash",  # deepseek-flash or kimi-k3.
            "api_key": "",
            "api_key_env": "",  # Blank chooses DEEPSEEK_API_KEY or MOONSHOT_API_KEY.
            "base_url": "",  # Blank chooses the provider's default endpoint.
            "max_tokens": 24000,
            "temperature": 0.0,
            "thinking": True,
            "reasoning_effort": "high",  # low, high, or max.
            "timeout_s": 120,
            "max_retries": 5,
            "retry_base_delay_s": 2,
            "retry_max_delay_s": 30,
        },
        "plugins": {
            "action_chunk": True,
            "affordance": True,
            "auto_release": True,
            "base": True,
            "decision_json": True,
            "decision_output": True,
            "decision_token": True,
            "extra_context": True,
            "mem_text": True,
            "observations": True,
            "proprioception": True,
            "recovery": True,
            "strict_token_retry": True,
            "subgoal": True,
            "variable_step": True,
            "whole_task_stage": True,
            "wrist_visibility": True,
        },
        "plugin_state": {},  # Each episode gets fresh plugin state.
        "extra_context": "",  # Task-independent instructions for planner and controller.
    }


def observe_context(
    context: dict, *, executor, observation: dict, client, task: str,
    subgoal: dict | None = None, stage_key: str = "", recent_moves=(), recovery=None,
    images=None, debug: bool = False, trace=None, stage_index=0, subgoals=(),
    target_in_wrist=None,
) -> dict:
    """Resolve live fields for one observation; keep every user-added context field."""
    from core.action_units import CAMERA_ATOMS, CONTROLLER_ATOMS, RT_ATOMS
    from core.sim.libero_observation import observation_images

    # Reuse prepared frames when planner and controller see the same observation.
    agentview, wrist = observation_images(observation, context) if images is None else images
    actions = context["semantic_eef_action"]

    current = {
        **context,
        "task": task,
        "stage_key": stage_key,
        "stage_index": stage_index,
        "subgoals": subgoals,
        "subgoal": subgoal if subgoal is not None else {},
        "recent_moves": list(recent_moves),  # Oldest first.
        "previous_direction": recent_moves[-1] if recent_moves else "NONE",
        "gripper_state": "CLOSED" if executor.gripper > 0 else "OPEN",
        "proprioception": read_proprioception(observation),
        "table_height_m": float(executor.env.env.workspace_offset[2]),
        "camera_frames": executor.camera_frames(),
        "base_frame_world": executor.base_frame_world().tolist(),  # Columns: FWD, RIGHT, UP.
        "recovery": recovery,
        "allowed_tokens": (
            CONTROLLER_ATOMS
            + (RT_ATOMS if actions["rotation_enabled"] else ())
            + (CAMERA_ATOMS if actions["camera_enabled"] else ())
        ),
        "agentview_image": agentview,
        "wrist_image": wrist,
        "client": client,
        "env": executor.env,
        "executor": executor,
        "observation": observation,
        "trace": trace,
        "debug": debug,
        "affordance": None,  # The affordance plugin fills this and annotates the images.
        "target_in_wrist": target_in_wrist,
        "response": None,
        "token": None,
        "success": False,
        "subgoal_done": False,
    }
    return current


def read_proprioception(observation: dict) -> dict:
    """Fields from the current observation returned by LIBERO reset/step."""
    import numpy as np

    return {
        "eef_pos": np.asarray(observation["robot0_eef_pos"]).tolist(),
        "eef_quat_xyzw": np.asarray(observation["robot0_eef_quat"]).tolist(),
        "gripper_width_m": float(np.abs(observation["robot0_gripper_qpos"]).sum()),
        "joint_pos": np.asarray(observation["robot0_joint_pos"]).tolist(),
    }
