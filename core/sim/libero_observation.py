"""Prepare simulator camera frames for model input and episode logging."""
import numpy as np
from PIL import Image

from core.record.images import prepare_view


def observation_images(obs: dict, cfg: dict):
    l = cfg["libero"]
    agent = np.array(obs["agentview_image"], copy=True)
    noise = l["perturbations"]["sensor_noise"]
    if noise["type"] != "none":
        # Use upstream noise functions on EVERY observation, including initial,
        # camera-only, and final frames; upstream set_init_state omits this step.
        from libero.libero.envs import env_wrapper
        agent = getattr(env_wrapper, noise["type"])(Image.fromarray(agent), severity=noise["severity"])
        agent = np.clip(agent, 0, 255).astype(np.uint8)
    agent = prepare_view(agent, rotation_degrees=l["agentview_rotation"])
    wrist = (prepare_view(obs["robot0_eye_in_hand_image"], rotation_degrees=l["wrist_rotation"])
             if l["use_wrist_image"] else None)
    return agent, wrist
