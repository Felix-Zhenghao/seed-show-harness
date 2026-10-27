"""Episode orchestration independent of the simulator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from harness.actions import AtomicAction, Decision
from harness.config import RunConfig, load_api_key
from harness.context import ContextUpdater, EpisodeContext
from harness.decision import DecisionLoop
from harness.domain import ControlStep, ExecutedAction
from harness.environments import create_environment
from harness.models.deepseek import DeepSeekClient
from harness.models.planning import HarnessPlanningClient
from harness.ports import SimulationEnvironment
from harness.serialization import Object
from harness.trajectory import TrajectoryStore
from plugins.controller import PluginController
from plugins.registry import create_registry


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    decisions: int
    end_reason: str
    trajectory_directory: str


class ActionExecutor:
    def __init__(
        self, environment: SimulationEnvironment, updater: ContextUpdater, trajectory: TrajectoryStore
    ):
        self.environment, self.updater, self.trajectory = environment, updater, trajectory

    def execute(self, decision: Decision, source: str = "controller") -> None:
        execution_id = uuid4().hex
        decision_id = self.updater.state.decision_id
        self.trajectory.record(
            "action_started",
            execution_id=execution_id,
            decision_id=decision_id,
            action=decision.action.value,
            rationale=decision.rationale,
            source=source,
        )

        def record_step(step: ControlStep) -> None:
            self.trajectory.record(
                "control_step", execution_id=execution_id, decision_id=decision_id, step=step
            )

        try:
            result = self.environment.execute(decision.action, record_step)
            if not result.completed:
                raise RuntimeError("Environment reported an incomplete action")
            action = ExecutedAction(
                execution_id,
                decision_id,
                decision.action.value,
                decision.rationale,
                source,
                len(result.control_steps),
            )
            self.trajectory.record("action_completed", decision_id=decision_id, action=action)
            self.updater.executed(action, result.observation)
            self.trajectory.observation(result.observation, decision_id, f"after {source} action")
        except BaseException as error:
            self.trajectory.record(
                "action_failed",
                execution_id=execution_id,
                decision_id=decision_id,
                error=str(error),
                source=source,
            )
            raise

    def release(self, rationale: str) -> None:
        self.execute(Decision(AtomicAction.RELEASE, rationale), "auto_release")


def run_episode(config: RunConfig, progress: Callable[[str], None] = print) -> EpisodeOutcome:
    config.validate()
    key = load_api_key(config.model)
    trajectory = TrajectoryStore(
        config.output_root, config.environment.suite, config.environment.task_id, (key,)
    )
    progress(f"Trajectory: {trajectory.directory}")
    trajectory.write_json("metadata.json", {"schema_version": 1, "config": config})
    environment: SimulationEnvironment | None = None
    client = DeepSeekClient(config.model, key, trajectory)
    decisions, success, reason = 0, False, "error"
    failure: Object | None = None
    try:
        environment = create_environment(config, trajectory.directory)
        observation = environment.reset()
        trajectory.write_json(
            "metadata.json", {"schema_version": 1, "config": config, "task": environment.metadata}
        )
        trajectory.observation(observation, 0, "initial")
        updater = ContextUpdater(
            EpisodeContext(environment.metadata, observation, trajectory.history_path), config.history_length
        )
        executor = ActionExecutor(environment, updater, trajectory)
        planner = HarnessPlanningClient(client, trajectory, config.model.dsh_executable)
        registry = create_registry(config, client, planner, executor.release)
        controller = PluginController(registry, updater, trajectory)
        loop = DecisionLoop(client, controller, updater, trajectory, config.max_output_corrections)
        reason = "decision_limit"
        for _ in range(config.max_decisions):
            if updater.state.observation.success:
                success, reason = True, "success"
                break
            controller.begin_decision(config.max_tool_calls_per_decision)
            if updater.state.observation.success:
                success, reason = True, "success"
                break
            decision = loop.decide()
            executor.execute(decision)
            decisions += 1
            progress(
                f"Decision {decisions}/{config.max_decisions}: {decision.action.value} — {decision.rationale}"
            )
            if updater.state.observation.success:
                success, reason = True, "success"
                break
    except KeyboardInterrupt:
        reason = "interrupted"
        failure = {"type": "KeyboardInterrupt", "message": "Interrupted by operator"}
    except Exception as error:
        reason = "error"
        failure = {"type": type(error).__name__, "message": str(error)}
        progress(f"Episode failed: {type(error).__name__}: {error}")
    finally:
        try:
            if environment is not None:
                environment.close()
        finally:
            client.close()
            trajectory.finish(
                {
                    "status": "finished",
                    "success": success,
                    "decisions": decisions,
                    "end_reason": reason,
                    "failure": failure,
                    "model_calls": client.model_call_count,
                }
            )
    return EpisodeOutcome(success, decisions, reason, str(trajectory.directory))
