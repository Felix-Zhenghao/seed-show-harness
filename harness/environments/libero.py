"""Original LIBERO adapter; all upstream/private simulator access is isolated here."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import yaml
from PIL import Image

from harness.actions import AtomicAction, interpret
from harness.config import EnvironmentConfig, RunConfig
from harness.domain import (
    CameraFrame,
    ControlStep,
    ExecutionResult,
    ImageAsset,
    Matrix3,
    Observation,
    RobotState,
    TaskMetadata,
    Vec3,
)


def task_catalog(config: EnvironmentConfig) -> dict[str, list[str]]:
    path = config.libero_root / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            return cast(dict[str, list[str]], ast.literal_eval(node.value))
    raise ValueError(f"No LIBERO task catalog in {path}")


def configure_libero(config: EnvironmentConfig, directory: Path) -> None:
    root = config.libero_root.resolve()
    package = root / "libero/libero"
    if not (package / "envs/env_wrapper.py").is_file():
        raise FileNotFoundError(f"Original LIBERO installation missing: {root}")
    if root.name.lower() == "libero-plus":
        raise ValueError("LIBERO-plus is not supported")
    if "libero.libero" in sys.modules:
        raise RuntimeError("Run each LIBERO environment in a fresh process")
    path_directory = directory / "libero_paths"
    path_directory.mkdir(parents=True, exist_ok=True)
    dataset_directory = path_directory / "datasets"
    dataset_directory.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "assets": str(package / "assets"),
        "datasets": str(dataset_directory),
    }
    (path_directory / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(path_directory)
    os.environ["MUJOCO_GL"] = (
        ("cgl" if sys.platform == "darwin" else "egl") if config.gl_backend == "auto" else config.gl_backend
    )
    sys.path.insert(0, str(root))


def _vector(value: Any) -> Vec3:
    return cast(Vec3, tuple(float(v) for v in value))


def _matrix(value: Any) -> Matrix3:
    return cast(Matrix3, tuple(_vector(row) for row in np.asarray(value).reshape(3, 3)))


class LiberoEnvironment:
    def __init__(self, config: RunConfig, directory: Path):
        self.config = config
        self._env: Any = None  # Upstream robosuite exposes an untyped runtime object.
        self._raw: dict[str, Any] = {}
        self._sequence = 0
        self._gripper = -1.0
        configure_libero(config.environment, directory)
        import libero.libero as libero_package
        from libero.libero import benchmark

        if (
            not Path(libero_package.__file__)
            .resolve()
            .is_relative_to(config.environment.libero_root.resolve())
        ):
            raise RuntimeError("LIBERO imports resolved outside the configured original checkout")
        settings = config.environment
        suite = benchmark.get_benchmark_dict()[settings.suite](task_order_index=settings.task_order_index)
        task = suite.get_task(settings.task_id)
        self._bddl_path = Path(suite.get_task_bddl_file_path(settings.task_id))
        self._initial_state_path = (
            settings.libero_root / "libero/libero/init_files" / task.problem_folder / task.init_states_file
        )
        self._metadata = TaskMetadata(
            "libero",
            settings.suite,
            settings.task_id,
            task.name,
            task.language,
            settings.seed,
            settings.initial_state,
            simulator_state_format="robosuite.MjSimState.flatten",
        )

    @property
    def metadata(self) -> TaskMetadata:
        return self._metadata

    def reset(self) -> Observation:
        import torch
        from libero.libero.envs import OffScreenRenderEnv

        if self._env is not None:
            raise RuntimeError("Create a fresh adapter for each episode")
        settings = self.config.environment
        # Original assets are trusted local benchmark files; torch 2.6 changed
        # the default loader, while the upstream helper does not specify it.
        states = np.asarray(torch.load(self._initial_state_path, weights_only=False))
        if states.ndim == 1:
            states = states[None, :]
        if not 0 <= settings.initial_state < len(states):
            raise ValueError(f"Initial state index outside [0, {len(states)})")
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(self._bddl_path),
            camera_heights=settings.image_size,
            camera_widths=settings.image_size,
            control_freq=settings.control_hz,
            hard_reset=False,
            ignore_done=True,
            horizon=1_000_000,  # Decision/intervention limits belong to the runner.
        )
        try:
            self._env.seed(settings.seed)
            # Avoid the wrapper's unbounded placement retry loop.
            self._env.env.reset()
            self._env.set_init_state(states[settings.initial_state])
            robot = self._env.robots[0]
            robot.controller.update(force=True)
            robot.controller.reset_goal()
            for _ in range(settings.settle_steps):
                self._env.step([0, 0, 0, 0, 0, 0, -1])
            robot.controller.use_delta = False
            self._base_body = self._env.sim.model.body_name2id(robot.robot_model.root_body)
            self._site = robot.eef_site_id
            self._metadata = TaskMetadata(
                **{**self._metadata.__dict__, "instruction": " ".join(self._env.language_instruction.split())}
            )
            return self.observe()
        except BaseException:
            self.close()
            raise

    def _robot_state(self) -> RobotState:
        data = self._env.sim.data
        return RobotState(
            _vector(data.site_xpos[self._site]),
            _matrix(data.site_xmat[self._site]),
            _matrix(data.body_xmat[self._base_body]),
            tuple(float(v) for v in self._raw["robot0_joint_pos"]),
            float(np.abs(self._raw["robot0_gripper_qpos"]).sum()),
            self._gripper,
            _vector(data.body_xpos[self._base_body]),
        )

    def observe(self) -> Observation:
        if self._env is None:
            raise RuntimeError("Environment has not been reset")
        self._raw = self._env.env._get_observations(force_update=True)
        self._sequence += 1
        rotation = self.config.environment.image_rotation
        images, cameras = [], []
        for name in ("agentview", "robot0_eye_in_hand"):
            pixels = np.ascontiguousarray(np.rot90(self._raw[f"{name}_image"], rotation // 90))
            images.append(ImageAsset.from_image(name, Image.fromarray(pixels)))
            index = self._env.sim.model.camera_name2id(name)
            cameras.append(
                CameraFrame(
                    name,
                    _vector(self._env.sim.data.cam_xpos[index]),
                    _matrix(self._env.sim.data.cam_xmat[index]),
                    float(self._env.sim.model.cam_fovy[index]),
                    rotation,
                )
            )
        return Observation(
            self._sequence,
            float(self._env.sim.data.time),
            tuple(images),
            tuple(cameras),
            self._robot_state(),
            bool(self._env.check_success()),
            self.simulator_state(),
        )

    def execute(self, action: AtomicAction, on_step: Callable[[ControlStep], None]) -> ExecutionResult:
        command = interpret(action, self._robot_state(), self.config.actions)
        self._gripper = command.gripper
        vector = command.vector()
        steps = []
        for index in range(command.control_steps):
            self._raw, _, done, _ = self._env.step(vector)
            success = bool(self._env.check_success())
            step = ControlStep(
                index, float(self._env.sim.data.time), vector, self._robot_state(), success, bool(done)
            )
            steps.append(step)
            on_step(step)
            if success:
                break
        return ExecutionResult(self.observe(), tuple(steps), True)

    def simulator_state(self) -> tuple[float, ...]:
        """Optional backend diagnostic for exact physical-state inspection."""
        return tuple(float(v) for v in self._env.get_sim_state())

    def close(self) -> None:
        if self._env is not None:
            env, self._env = self._env, None
            env.close()
