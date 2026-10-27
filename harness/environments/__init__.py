"""Environment factories; additional simulators implement SimulationEnvironment."""

from collections.abc import Callable
from pathlib import Path

from harness.config import RunConfig
from harness.ports import SimulationEnvironment

Factory = Callable[[RunConfig, Path], SimulationEnvironment]


def create_environment(config: RunConfig, directory: Path) -> SimulationEnvironment:
    from harness.environments.libero import LiberoEnvironment

    factories: dict[str, Factory] = {"libero": LiberoEnvironment}
    try:
        factory = factories[config.environment.name]
    except KeyError as error:
        raise ValueError(f"Unknown environment: {config.environment.name}") from error
    return factory(config, directory)
