# Show-Harness · single-arm LIBERO

Run a vision-language agent with one simulated Franka Panda. The entry point is
`run.sh`; settings and prompt inputs are defined in [`context.py`](context.py).

```bash
./run.sh --check       # Validate settings and selected task assets
./run.sh --list-tasks  # List original task indices and names
./run.sh               # Run the configured task(s)
```

The launcher works from any directory. Edit `context.py` to configure a run;
shared dependency and output paths use the parent repository root defined there.
The setup commands below run from the parent `Show-Harness/` directory.

## Setup

Keep the original LIBERO and LIBERO-Plus checkouts at the paths below, or update
`libero.original_root` and `libero.plus_root` in the configuration. These are external
dependencies, excluded from this repository's Git changes and lint checks.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus
```

Download the LIBERO-Plus scene assets using its checkout's instructions, placing the
assets at `third_party/LIBERO-plus/libero/libero/assets/`. The original checkout supplies the
canonical tasks and saved initial states. Install `uv` and ImageMagick first
(`brew install imagemagick` on macOS), then create the configured Python environment:

```bash
bash seed_harness/scripts/setup_libero.sh third_party/LIBERO-plus
```

The installer uses Python 3.10, native macOS PyTorch/CGL or Linux CUDA/EGL, and installs
both simulator and harness dependencies. `runtime.python` selects the interpreter.
`runtime.gl_backend` can override the rendering backend.

If macOS reports `MagickWand shared library not found` despite ImageMagick being
installed, export its path before launching: `export MAGICK_HOME="$(brew --prefix imagemagick)"`.

Set the API key for the model selected in `model.name`: `DEEPSEEK_API_KEY` or
`MOONSHOT_API_KEY`. You can export it or copy `configs/secrets.env.example` to the
ignored `configs/secrets.local.env`. `model.api_key` and `model.api_key_env` are also
supported. Saved configuration and metadata redact credential fields.

## Runtime

The agent plans subgoals, selects semantic actions from scene and wrist images, and
executes them through LIBERO's Cartesian controller. LIBERO's success predicate decides
episode success. Translation, rotation, camera movement, plugin switches, model options,
and debugging are configured directly in `context.py`.

- `tasks.mode`: a single task, a bundle, or the configured benchmark suites.
- `libero.perturbations`: automatically selects LIBERO-Plus when any perturbation is enabled.
- `plugins`: planning, proprioception, action memory/chunks, variable steps,
  grasp recovery, automatic release, and affordance markers with a VLM fallback.
- `runtime.debug`: saves exact model inputs/replies, simulator states, and `trajectory.html`.

Runs are written to `runs/libero/<timestamp>/`, including results, images, plans,
per-step records, and a video. See [launch details](docs/libero-launch.md) for
perturbation constraints, action coordinates, and log contents.

## Code layout

| Path | Responsibility |
| --- | --- |
| `context.py` | Explicit constants, live simulator values, and callable context providers. |
| `core/sim/` | Validation, task assets, simulator setup, model dialects, episode loop. |
| `core/vlm/` | Hosted model transport, response parsing, and controller calls. |
| `core/prompting/` | Direct builder calls, prompt order, and text/image output values. |
| `core/record/` | Images, videos, and debug records. |
| `interpreters/libero_atomic_controller.py` | Franka Cartesian, gripper, and camera actions. |
| `plugins/` | Each plugin owns its `build.py`, wording, model/API calls, and runtime behavior; [authoring guide](plugins/README.md). |
| `scripts/setup_libero.sh` | Simulator environment installation. |
| `tests/` | Runtime unit tests and optional MuJoCo integration tests. |

The shared opening lives in [plugins/base/prompt.txt](plugins/base/prompt.txt).
Its instruction section contains action tokens and movement/rotation sizes; its
`[state]` section contains the task, intended grasping part, and subgoal. Its `build.py`
fills those values directly from context. The instructions come first in planner
and controller prompts. EEF actions follow the wrist's current
orientation; camera actions follow Agentview's current displayed frame.

Prompts place reusable instructions before task/subgoal data and per-step feedback,
then append the current Agentview and Wristview images in that order. The base,
variable-step, and memory plugins expose `build_instructions(context)` and `build_state(context)`
so the composer can group their rules before live data; `build(context)` still returns
the complete standalone text. Affordance calls also put rules (and the simulator
catalog for selection) before task, stage, part, and view details.

This gives the provider a stable prefix to cache without reusing old observations or
model answers. [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache)
is automatic and best effort; repeated prefixes may need several requests to warm up.
Compare `usage.prompt_cache_hit_tokens` with `usage.prompt_tokens` in saved
`calls/*_response.json` files to measure the input-token hit rate. This optimization
affects input cost; output/reasoning tokens still incur their usual cost.

## Add a plugin

1. Create `plugins/task_hint/prompt.txt` with `Task reminder: ${task}` and a `build.py`:

   ```python
   from pathlib import Path
   from string import Template

   def build(context):
       if context["plugins"].get("task_hint") is not True:
           return ""
       text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
       return Template(text).substitute(task=context["task"]).strip()
   ```

2. Add `"task_hint": True` to `plugins` in [context.py](context.py).
3. In [composer.py](core/prompting/composer.py), import
   `from plugins.task_hint import build as task_hint` and add `task_hint.build` to the
   desired `compose(...)` call.

The builder returns the complete prompt. The composer only calls builders in order and
joins their results. Static wording still has a small `build(context)`; image-only
plugins can omit `prompt.txt`.

Keep all plugin behavior in its `build.py`, including API/model calls, response parsing,
and any runtime functions. Share inputs and results through `context`; use
`context["plugin_state"]["task_hint"]` for state that persists within an episode.

Edit `create_context()` for constants and switches, and `observe_context()` for live
inputs; both are in [context.py](context.py). New fields reach builders directly.
For example, the affordance builder reads `env` and `client`, annotates the image fields,
and stores its result in `context["affordance"]`. See the [plugin guide](plugins/README.md)
for runtime examples and text/image composition.

The affordance plugin first selects a named simulator part. If none matches, it asks
the VLM to locate the contact point, draws a candidate dot, and verifies/corrects it.
It tracks the point in both current views and omits hidden parts. All of this lives
in `plugins/affordance/build.py` and `prompt.txt`; enable it with `plugins.affordance`.
The fallback adds model calls for pointing, verification, and tracking; debug records
identify whether each annotation came from the simulator or VLM.

## Checks

The normal tests require only the harness dependencies and `pytest`; they make no
model calls and need no downloaded simulator assets.

```bash
python -m pip install -r seed_harness/requirements/requirements.txt pytest ruff
ruff check .
python -m pytest -q
```

With the simulator environment and assets installed:

```bash
LIBERO_SIM_TESTS=1 third_party/LIBERO-plus/.venv/bin/python -m pytest -q seed_harness/tests/test_libero_sim.py
```

These integration checks exercise real physics, camera actions, perturbations,
affordance markers, automatic release, and logging using scripted model replies.
