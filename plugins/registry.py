"""Composition root for the enabled plugin pool."""
from collections.abc import Callable

from harness.config import RunConfig
from harness.models.deepseek import CompletionClient
from plugins.action_mem.build import PLUGIN as action_mem
from plugins.affordance.build import create_plugin as affordance
from plugins.auto_release.build import create_plugin as auto_release
from plugins.base_prompt.build import create_plugin as base_prompt
from plugins.contracts import PluginSpec
from plugins.subgoal.build import PlanningClient
from plugins.subgoal.build import create_plugin as subgoal


def create_registry(
    config: RunConfig, client: CompletionClient, planner: PlanningClient, release: Callable[[str], None]
) -> dict[str, PluginSpec]:
    pool = {
        plugin.name: plugin
        for plugin in (
            action_mem,
            subgoal(planner),
            affordance(client),
            auto_release(release, config.actions.empty_gripper_width_m),
        )
    }
    unknown = set(config.plugins) - {*pool, "base_prompt"}
    if unknown:
        raise ValueError(f"Unknown enabled plugins: {sorted(unknown)}")
    pool["base_prompt"] = base_prompt(config.actions)
    return {name: pool[name] for name in config.plugins}
