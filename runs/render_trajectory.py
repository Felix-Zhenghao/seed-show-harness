#!/usr/bin/env python3
"""Usage: python runs/render_trajectory.py EPISODE_DIRECTORY_OR_EVENTS_JSONL."""

import sys
from pathlib import Path

# Allow use from any working directory without installing the harness package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.trajectory_viewer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
