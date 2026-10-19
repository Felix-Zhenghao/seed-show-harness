"""Optional local API credentials; launch settings live only in context.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from context import ROOT


DEFAULT_SECRETS_ENV = ROOT / "configs" / "secrets.env"


def load_secrets_env(path: str | Path | None = None, *, override: bool = False) -> Dict[str, str]:
    """Load ``KEY=VALUE`` pairs from ``configs/secrets.env`` into ``os.environ``.

    Dependency-free dotenv parser (no python-dotenv): blank lines and ``#`` comments
    are skipped, an optional ``export`` prefix is allowed, surrounding single/double
    quotes are stripped, and whitespace around ``=`` is tolerated (``KEY = "v"``).
    A missing file is not an error (returns ``{}``). Existing environment variables
    are preserved unless ``override=True``, so a value exported in the shell wins.
    With the default path, an optional gitignored per-machine overlay
    ``configs/secrets.local.env`` is read on top (its keys win over the shared file).

    Returns the mapping that was parsed from the file (regardless of whether it was
    written to the environment), so callers can report which keys are available.
    """
    if path is None:
        # Default flow: the shared file, then an optional per-machine overlay
        # (gitignored, same format). Overlay keys win over the shared file;
        # shell-exported variables still win over both.
        parsed = _parse_env_file(DEFAULT_SECRETS_ENV)
        parsed.update(_parse_env_file(DEFAULT_SECRETS_ENV.with_name("secrets.local.env")))
        for key, value in parsed.items():
            if override or key not in os.environ:
                os.environ[key] = value
        return parsed
    env_path = Path(path)
    parsed = _parse_env_file(env_path)
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not env_path.is_file():
        return parsed
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
    return parsed
