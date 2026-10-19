#!/usr/bin/env bash
# Launch from any working directory. Settings: context.py. Options: --check, --list-tasks.
set -euo pipefail
RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_ROOT"
export SHOW_HARNESS_ROOT="${SHOW_HARNESS_ROOT:-$(dirname "$RUN_ROOT")}"
RUN_BOOTSTRAP="${SHOW_HARNESS_ROOT}/.venv/bin/python"
[[ -x "$RUN_BOOTSTRAP" ]] || RUN_BOOTSTRAP="${SHOW_HARNESS_ROOT}/third_party/LIBERO-plus/.venv/bin/python"
[[ -x "$RUN_BOOTSTRAP" ]] || RUN_BOOTSTRAP=python3
export PYTHONPATH="${RUN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_PYTHON="$("$RUN_BOOTSTRAP" -c 'from context import create_context; print(create_context()["runtime"]["python"])')"
[[ -x "$RUN_PYTHON" ]] || {
  echo "Configured Python not found: $RUN_PYTHON. Run bash seed_harness/scripts/setup_libero.sh third_party/LIBERO-plus first." >&2
  exit 2
}
exec "$RUN_PYTHON" -m core.sim.libero_runner "$@"
