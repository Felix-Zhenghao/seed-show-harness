# Standalone plugins

Each plugin owns its wording and implementation:

```text
plugins/<name>/
  build.py     # build(context), API/model calls, parsing, and runtime behavior
  prompt.txt   # Wording, when the plugin produces text
```

Every plugin has a normal `build(context)` function. It checks its switch first,
loads its own text, fills the blanks, and returns finished text or `PromptContent`.
An empty string contributes nothing. An image-only plugin can omit `prompt.txt`.

[composer.py](../core/prompting/composer.py) imports builders directly and calls them
in an explicit order. `compose(context, *builders)` only calls those functions and
joins their completed results. Ordinary Python imports resolve plugins.

`base.build_instructions` comes first in planner and controller prompts, including controller retries.
It owns the camera introduction, action explanations,
grasp/release/completion rules, task, and subgoal. These replace the former `world_axes`,
`rotation_actions`, `camera_actions`, `direction`, `gripper`, `completion`, `task`, and
`stage` plugins. Edit [base/prompt.txt](base/prompt.txt) for wording and
[base/build.py](base/build.py) for its substitutions. Enable it with `plugins.base`;
rotation and camera actions are enabled in `semantic_eef_action`.
The base text has an instruction section for action tokens and movement/rotation sizes,
followed by a `[state]` section for task, affordance, and subgoal data. `build(context)`
returns both sections; `build_instructions(context)` and `build_state(context)` let
the composer place other reusable instructions between them. Substitution remains
inside the plugin and never reinterprets inserted text.

Keep reusable instructions before changing values to improve provider prefix-cache
reuse. The controller groups base, extra-context, wrist, action-plan, coarse-step, and memory
rules before task/stage data, measured state, history, and feedback. The planner puts
its output instructions before task/stage data too. `controller_prompts(context)`
builds the shared instructions and live state once, then returns the JSON decision
and token fallback. JSON output and step-size rules go between shared instructions
and live state, making them part of the reusable prefix. The token fallback uses the
same completed sections without JSON-only rules, then appends its token contract.
Retries never repeat plugin side effects. Model transport appends current camera
images after the prompt, preserving Agentview/Wristview order and any image positions
inside plugin content.
Do not memoize live builder results: affordance markers and observations must refresh.

## Add a plugin

1. Write `plugins/task_hint/prompt.txt`:

   ```text
   Task reminder: ${task}
   ```

2. Write `plugins/task_hint/build.py`:

   ```python
   from pathlib import Path
   from string import Template

   def build(context):
       if context["plugins"].get("task_hint") is not True:
           return ""
       text = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
       return Template(text).substitute(task=context["task"]).strip()
   ```

3. Add `"task_hint": True` to `plugins` in [context.py](../context.py).
4. Import `from plugins.task_hint import build as task_hint` in the composer and add
   `task_hint.build` to the desired `compose(...)` call.

The example uses Python's `string.Template`; each builder can choose its own filling
logic. `${slot}` inserts a value and `$$` writes a literal dollar sign. JSON braces
need no escaping. Some bundled plugins keep named text sections in `prompt.txt` and
read those sections locally in `build.py`.

For a static prompt, the builder simply checks its switch and returns the file text.
Call `task_hint.build(context)` directly to exercise the plugin on its own.

## Share data through context

[context.py](../context.py) lists constants in `create_context()` and live simulator
inputs in `observe_context()`. Add a field there and read it directly in a builder.
Plugin-specific API/model calls and parsing belong in that plugin's `build.py`.

Inputs include the current `task`, `subgoal`, `recent_moves`, `proprioception`,
`camera_frames`, `base_frame_world`, images, `env`, `executor`, and `client`. Every builder in a composition
receives the same dictionary, so earlier builders can write results for later builders.
Custom fields also survive model transport and retries.

Persistent plugin state goes in `context["plugin_state"]["your_name"]`. Each episode
starts with a fresh state dictionary; observation contexts share it within that episode.
Keep mutable runtime state out of module globals and keep imports free of API/model calls.

Examples of plugin-owned behavior:

| Plugin | Implementation and shared results |
| --- | --- |
| `base` | Separates configured action rules from task/subgoal state; camera descriptions, frame explanations, and examples are written directly in `prompt.txt`. |
| `affordance` | Selects a simulator part or falls back to VLM point/verify/track calls. Updates both image fields, writes `context["affordance"]`, and returns marker guidance. The simulator API and visual fallback live in the same `build.py`. |
| `subgoal` | Builds planner instructions; `plan(context)` owns model calls, retries, parsing, and stage merging. It consumes `planner_prompt` and `fallback_stage`, and writes planner diagnostics. |
| `wrist_visibility` | Builds the visibility instruction; `read_response(context)` writes `target_in_wrist`. |
| `action_chunk` | Builds planning instructions, parses `chunk_plan`, and owns the queued moves in `plugin_state`. |
| `variable_step` | Builds coarse-movement guidance; `step_m(context)` reads the measured height and wrist result. |
| `recovery`, `auto_release` | Own grasp checks, reopening, and their prompt feedback. |
| `mem_text` | Owns history retention and its prompt order. |

Runtime functions such as `plan`, `before_decision`, and `after_step` are plain functions
in the same `build.py`. The runner calls them directly. Plugins exchange data through
context and do not import one another.

For images, return `PromptContent(("History:", PromptImage(image), "Continue."))` using
[content.py](../core/prompting/content.py). Text and images retain their order through
composition and retries. The shared content types only represent output; filling and
image selection remain inside the plugin.
