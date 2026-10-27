"""Command-line entry points for the standalone seed harness."""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

from harness.config import load_api_key, load_config
from harness.environments.libero import task_catalog
from harness.runner import run_episode
from harness.serialization import dumps


def main() -> int:
    parser = argparse.ArgumentParser(description="Context-oriented simulation robot harness")
    parser.add_argument("command", choices=("run", "check", "list-tasks"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--suite")
    parser.add_argument("--task", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    environment_changes = {}
    if args.suite is not None:
        environment_changes["suite"] = args.suite
    if args.task is not None:
        environment_changes["task_id"] = args.task
    config = dataclasses.replace(
        config, environment=dataclasses.replace(config.environment, **environment_changes)
    )
    if args.max_decisions is not None:
        config = dataclasses.replace(config, max_decisions=args.max_decisions)
    if args.output_root is not None:
        config = dataclasses.replace(config, output_root=args.output_root.resolve())
    config.validate()
    if args.command == "list-tasks":
        print(dumps(task_catalog(config.environment)))
        return 0
    if args.command == "check":
        catalog = task_catalog(config.environment)
        load_api_key(config.model)
        if not shutil.which(config.model.dsh_executable):
            raise FileNotFoundError("DeepSeek Harness is not installed")
        print(
            dumps(
                {
                    "configuration": "valid",
                    "credentials": "available",
                    "dsh": "available",
                    "suite": config.environment.suite,
                    "task": catalog[config.environment.suite][config.environment.task_id],
                }
            )
        )
        return 0
    outcome = run_episode(config)
    print(dumps(outcome))
    return 1 if outcome.end_reason in ("error", "interrupted") else 0


if __name__ == "__main__":
    raise SystemExit(main())
