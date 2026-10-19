# LIBERO with an explicit Python context

`plugins.auto_release=True` reopens an empty closed gripper after each
action, including a later slip. Its implementation in `plugins/auto_release/build.py` uses `harness.empty_width_m` and works with
`recovery=False`; recovery separately validates grasps and rewinds the plan.
Setting `auto_release=False` also suppresses recovery's automatic reopening.
Explicit model `RELEASE` actions still work.

`plugins.affordance=True` first tries simulator parts. Once per spatial stage,
the VLM writes a single
`sim.get_object_part(object_name='...', part_name='...')` expression using a
catalog from the loaded environment. A restricted interpreter dispatches that
read-only API. MuJoCo supplies the live site/mesh/body reference center; the
plugin projects a red point into both controller views on each decision. Markers
follow moving objects, articulated parts, and both cameras. The controller prompt
identifies the marked grasp part or destination region. The marker radius is
`harness.affordance_dot_radius_px`. Debug logs include the API query, return value,
world position, projected pixels, and exact marked controller images.

These are privileged ground-truth reference centers, not optimized grasp poses.
They can overlay occluded parts; points outside the image or behind the camera
are omitted. If a requested part is absent from the catalog or selection fails,
the same plugin falls back to visual pointing:

1. Locate one visible contact point on a normalized `[y, x]` grid (0–1000).
2. Draw a candidate dot and ask the VLM to confirm, correct, or reject it.
3. Verify/correct the point on each new controller observation in both views.
   Agentview can move too. The current marked Agentview is a reference for the wrist.

Hidden parts, invalid coordinates, and failed calls produce no dot in that view;
the next observation can try again. Three unchanged tracking confirmations trigger
fresh localization to avoid repeatedly confirming a stale dot. A new stage clears
the selected contact. Rebuilding the same observation reuses its completed result.

The fallback uses the configured `context["client"]` and adds model calls: up to
one locate plus one verify per available view initially, then one tracking call per
visible view, with fresh localization after loss or stale tracking. It is adapted
from the official [point/verify/track implementation](https://github.com/showlab/Show-Harness/blob/101bd59d3413d32a85d3afa1164f62ddf31f99da/plugins/affordance/agent.py).
All runtime logic remains in `plugins/affordance/build.py`; prompts remain in its
`prompt.txt`. VLM markers are visual estimates, and controller guidance says so.
A model can incorrectly confirm its own dot; `verified=true` records its judgment,
not simulator validation of the contact point.
`context["affordance"]["source"]` is `simulator` or `vlm`; debug traces record the
operation, view, model replies, and resulting pixels. The
[sim-affordance skill](../skills/sim-affordance/SKILL.md) describes
the API and how to write queries and simulator-side adapters.

Edit the root [`context.py`](../context.py), then run:

```bash
./run.sh --check       # Validate settings and selected BDDL/state assets; no API calls.
./run.sh --list-tasks  # Original suite task indices and full names; no simulation.
./run.sh               # Execute the configured selection.
```

The launcher works from any directory. `create_context()` in `context.py` explicitly
lists every launch setting, and `observe_context()` resolves live prompt inputs.
Shared paths use the parent project root (or SHOW_HARNESS_ROOT). The generated `libero_paths/config.yaml` inside
each run is LIBERO's internal path cache; no user-maintained YAML is loaded.

Install the dependencies with `bash seed_harness/scripts/setup_libero.sh third_party/LIBERO-plus`. Keep the
original [LIBERO checkout](https://github.com/Lifelong-Robot-Learning/LIBERO) at
`third_party/LIBERO`, and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) at
`third_party/LIBERO-plus`, or change the two root paths in `context.py`. LIBERO-Plus needs its
[downloaded assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main),
including the added objects, textures, and scenes. See [setup](../README.md#setup)
for installation details. The configured Python must contain the simulator dependencies.

Settings are Python dictionary entries in `create_context()`. The main choices are:

| Setting | Effect |
| --- | --- |
| `tasks.mode="single"` | Run `tasks.task` in `tasks.suite`. |
| `tasks.mode="bundle"` | Run `tasks.task_ids` in that suite; `[]` runs the entire suite. |
| `tasks.mode="benchmark"` | Run every task in `tasks.benchmark_suites`, with the selected perturbation settings. The default five suites contain 130 original tasks. |
| `tasks.episodes`, `tasks.initial_state` | Use consecutive saved initial states; an unavailable state is an error, never silently wrapped. |
| `runtime.debug=True` | Save the full model/agent trajectory and `trajectory.html` for inspection. |
| `model.name` | `deepseek-flash` or `Kimi-K3`. |

Supply the selected model's key in `model.api_key`, or export `DEEPSEEK_API_KEY` /
`MOONSHOT_API_KEY`. The existing gitignored `configs/secrets.local.env` credential store
is also supported. A literal `model.api_key` value takes precedence over the named environment
variable. Credentials are removed from saved configuration and metadata; keep keys out
of commits. Endpoints are selected from the model name unless `model.base_url` is set.
For Kimi accounts using the China endpoint, set `https://api.moonshot.cn/v1` explicitly.

[DeepSeek Flash](https://api-docs.deepseek.com/guides/vision) accepts image input and can
disable provider thinking. [Kimi K3](https://platform.moonshot.ai/docs/guide/kimi-k3-quickstart)
always thinks, uses `max_completion_tokens`, and fixes its sampling parameters. Its
output budget includes reasoning, so increase both model and planner budgets if the
provider returns no final answer.

## Perturbations

With every perturbation at its default, the runner imports **original LIBERO** and uses
its original BDDL and saved initial states. Enabling a perturbation selects LIBERO-Plus
automatically. The original task IDs remain stable in both modes.

Camera orbit, camera orientation, distance, robot initial pose, instruction, lighting,
textures, object layouts, and each upstream sensor-noise family have named fields.
The runner generates any required filename encoding inside the run directory; source
checkouts are never rewritten. Camera distance scaling is the upstream distance change,
not a lens FOV change. Lighting variant IDs retain upstream's joint intensity, color,
direction, and shadow presets. Numeric variant IDs are upstream asset selections, not
random seeds.

Layout, language, appearance, camera, robot-pose, and noise perturbations can be combined.
Two upstream presets overlap and are explicitly mutually exclusive:

- `background_variant` includes a table texture, so choose it or `table_texture_variant`.
- Distractor and displacement layouts have separate saved states, so choose
  `distractor_variant` or `position_level`/`position_sample`.

Layout variants have one saved state: set `tasks.episodes=1` and `initial_state=0`.
Many Plus asset variants are provided only for the four ten-task suites; select those
in `benchmark_suites` when using those variants. Missing variants fail before model
calls. `benchmark` applies the configured settings to each base task; it does not expand
into the upstream 10,030-variant robustness sweep.

Robot pose perturbations are reapplied **after** loading object state, so restoration
cannot erase them. Sensor noise affects every external-camera input, including the first
observation and observations after camera actions. Wrist images follow upstream's clean
wrist behavior.

## Semantic actions

The existing planner, controller role, memory, proprioception, variable step, action
chunk, recovery, and video logger are reused. The supported switches for this runner are
listed in `plugins`. Required fields and value ranges are validated; additional
context fields are allowed so plugins can introduce their own data.
Every plugin lives in `plugins/<name>/build.py`, including its prompt building,
API/model calls, response parsing, and runtime behavior. Wording lives beside it in
`prompt.txt`. Each builder checks `context["plugins"][name]` and returns a completed
prompt. The composer calls builders directly and joins their results; plugins share
data and per-episode state through context.
See the [plugin guide](../plugins/README.md). `extra_context` adds instructions
to both model roles. The launcher and all feature switches use the object in `context.py`.

| Tokens | Meaning |
| --- | --- |
| `MV_FWD/BACK`, `MV_RIGHT/LEFT`, `MV_UP/DOWN` | Translations along robot base +X/-X, +Y/-Y, +Z/-Z. |
| `RT_ROLL_LEFT/RIGHT` | Positive/negative rotation about base FWD (+X) through the TCP. |
| `RT_PITCH_FWD/BACK` | Negative/positive rotation about base RIGHT (+Y) through the TCP. |
| `RT_YAW_CW/CCW` | Positive/negative rotation about base UP (+Z) through the TCP. |
| `CAM_RIGHT/LEFT`, `CAM_UP/DOWN` | Translate the external camera along base +Y/-Y, +Z/-Z. |
| `CAM_FWD/BACK` | Translate the external camera along base +X/-X. |
| `CAM_YAW_CW/CCW` | Rotate the camera in place positively/negatively about base UP (+Z). |

All three action families share the robot base frame. FWD/RIGHT/UP are the +X/+Y/+Z
axes of `env.robots[0].robot_model.root_body`, read from the simulator's body rotation
and exposed as `context["base_frame_world"]` (world-space columns). This replaces the
former wrist-attached `eef_frame_world` field. The base axes align with world axes in
normal LIBERO scenes, but a rotated base is handled explicitly. Wrist orientation,
camera orientation, and displayed image rotation never redefine action directions.
For example, after a 90-degree wrist or camera yaw, both `MV_FWD` and `CAM_FWD` still
translate along base +X. After tilting the wrist, `MV_UP` remains upward for an upright
base.

Rotation signs preserve the existing EEF convention and follow the right-hand rule
about the named base axis. Yaw CW/CCW is viewed looking along base UP, from below
toward above; CW is positive about +Z. EEF rotations pivot at the current TCP, and
camera yaw pivots at the current camera position, not at the base origin.
Camera targets are converted through the camera parent's world pose, including for
non-world parents. Camera actions refresh images without advancing physics.
The [base plugin](../plugins/base/prompt.txt) explains these conventions and begins
both planner and controller prompts. Variable-step and action-chunk instructions
also use the same base axes.

EEF commands target the gripper **site** controlled by OSC, not the differently oriented
hand-body quaternion in LIBERO observations. Each semantic motion sets one absolute
pose target and holds it for `semantic_eef_action.sim_steps`; increasing that count gives the
controller more settling time and does not multiply the requested displacement.
Translations preserve the chosen tilt. Actual achieved motion may differ under contact
or joint limits; the trajectory records measured poses and the commanded targets.

Controller JSON can override a single action's magnitude with `distance_m` for `MV_*`
or `angle_deg` for `RT_*`. Usually omit sizes; provide one when the default is unsuitable,
such as during precise alignment. Magnitudes are finite, positive numbers; the token
determines direction. Keep action selection in `reasoning`, and explain a size change
separately in `step_size_reasoning` only when supplying an override:

```json
{"decision":"MV_RIGHT","distance_m":0.008,"reasoning":"The gripper is just left of the target.","step_size_reasoning":"Use a smaller step for final alignment."}
```

Omitted fields or JSON `null` retain the current fine/coarse translation selection and
configured rotation angle. Override bounds are inclusive and configurable under
`semantic_eef_action`: `min_distance_m`/`max_distance_m` (defaults 0.0001–1 m) and
`min_angle_deg`/`max_angle_deg` (defaults 0.01–180 degrees). These bounds can be tightened
within the existing motion configuration limits; they apply only to model overrides.
Invalid or inapplicable fields produce a diagnostic and fall back to the default.
Sizes apply to one decision only. Any non-null size field, even an invalid one, disables
action chunking and clears queued moves. Recovery and token-only retries use defaults.
Memory text continues to contain only past semantic action tokens.

`DONE` advances a subgoal. Episode success always comes from LIBERO's success predicate;
the final model `DONE` alone cannot report a successful benchmark episode.

## Logs and checks

Each launch creates `runs/libero/<timestamp>/` with redacted `context.resolved.json` and
`results.json`. Individual episodes contain the existing analysis video, images,
`steps.json`, plan, metadata, and summary. Each step's `step_size` record contains the
requested fields, effective translation/rotation magnitude, `model` or `default` source,
and validation diagnostics. Its `measured` fields report TCP displacement in world
coordinates, displacement magnitude, and relative rotation angle from the controlled
site before/after the semantic action, before any automatic release. `vlm.c.why` remains
the action reasoning; `vlm.c.step_size_reasoning` holds the separate size explanation.
Debug mode additionally writes:

- `trajectory.html`: chronological observations, model inputs/replies, actions, and result.
- `calls/`: exact prompts and ordered image blocks, lossless input images, raw provider
  responses (including reasoning when returned), and usage/timing information.
- `trace.jsonl`: the corresponding structured events, including parse retries.
- `state_*.npz`: simulator state and camera pose before each decision, plus final state.

Errors and interruptions write a summary and close the simulator/video. API errors stop
execution instead of becoming the generic controller's fallback movement. Debug mode
records what the provider returns; it cannot expose reasoning the provider does not send.

```bash
third_party/LIBERO-plus/.venv/bin/python -m pytest -q
LIBERO_SIM_TESTS=1 third_party/LIBERO-plus/.venv/bin/python -m pytest -q seed_harness/tests/test_libero_sim.py
```

The opt-in simulation checks use fixed tasks in `libero_spatial` and `libero_goal`,
with original and composed perturbation settings, and make no model calls. For a short live test, select that single
task, enable debug, and temporarily reduce `harness.max_steps` (for example to 8).
