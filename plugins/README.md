# Robot harness plugins

These plugins run through the shared controller in `controller.py`. The `subgoal`
function invokes a DeepSeek Harness planning agent; `affordance` invokes a
DeepSeek Flash VLM with all current camera views and annotates one requested view.

## Three independent declarations

| Property | Values | Meaning |
| --- | --- | --- |
| Activation | every_decision, model_call, conditional | When a handler runs. |
| Result lifetime | decision, until_next_call | How long its latest result is reusable. |
| Dependencies | Tool names with argument builders | Information needed before execution. |

The controller is instantiated per episode. There is one current result per plugin
name, not a cache per argument combination. A valid explicit invocation always runs
again; only prerequisite resolution reuses an existing result.

An until_next_call result survives decision boundaries and dependency updates. Its
next invocation retires it after argument validation, before running prerequisites
or the handler. Failed replacement leaves no active result. The previous data is
available to the handler as historical context. It never leaks into another episode.

A decision result is tied to the decision, observation revision, and dependency
versions that produced it. Updating a dependency invalidates decision-scoped consumers
transitively. Propagation stops at an unchanged until_next_call result. This preserves
the literal persistent lifetime rather than introducing a hidden refresh rule.

## Ownership and adding plugins

- contracts.py defines the common interface and native function-tool declarations.
- Each plugin's build.py owns its handler, description, input/output schemas, and
  dependency argument mappings. Helper instructions live in prompt.txt.
- registry.py selects the enabled pool. Adding a tool does not require changing the
  central controller or repeating its definition in the base prompt.
- controller.py validates and resolves the dependency graph, manages result lifetimes,
  assembles context, and enforces the shared call budget.

Each dependency edge maps the consumer's arguments and current context to the
prerequisite's own argument schema. A mapping constructs arguments; it never invokes
another plugin. Handlers receive resolved PluginResult values in dependency_results.
They return structured data and optional context blocks instead of injecting them.
Future tools may use masks, object identifiers, numbers, or other arguments; prompt
strings and point coordinates belong only to the example tools.

Dependency edges connect model-call information tools. Automatic context builders
and conditional interventions are scheduled separately, so resolving an information
request cannot force an intervention. Unknown, disabled, duplicate, or cyclic
dependencies are configuration errors. Shared prerequisites are evaluated once while
their result remains valid.

## Decision sequence

1. Start with the latest observations, expire decision results, reset the call budget,
   and include existing until_next_call results.
2. Evaluate conditional hooks once in enabled-registry order. Record actual completed
   interventions and refresh observations when a result reports observations_changed.
3. Run every_decision builders against the resulting state. Present stable instructions,
   the task, current subgoal, observations, robot pose, and previous actions with their
   rationale. An active annotation replaces its matching observation image
   in model input; originals and retired annotations are not replayed alongside it.
4. Ask the model for one native tool call or one final atomic action and brief rationale.
   For a call, resolve missing prerequisites, inject each result immediately, then run
   the requested tool. No additional model turn is required between those executions.
5. Return the requested call's native tool response and ask the model again. On a final
   action, validate the existing two-field format and return it to the host for execution.

The default permits unlimited tool calls; a per-decision budget can be configured. Reserve a slot for
each explicit or prerequisite invocation before resolving its children. Invalid
arguments, unknown tool names, and failed invocations consume their reserved slot;
cached reads and automatic hooks do not. A parent whose prerequisite fails or cannot
fit the remaining budget does not execute, even though its attempt was reserved.
Successful prerequisites remain stored when a later dependent fails. Tools are
disabled at zero. Malformed final responses are rejected without execution. Native
tool calls take precedence over accompanying assistant text; an action is selected
only from a response without tool calls.

## Native tool calling and logging

The transport uses DeepSeek's documented
[Responses API](https://api-docs.deepseek.com/zh-cn/api/create-response/).
`tool_definitions` supplies the API's `tools` parameter with strict function schemas.
The system prompt describes when to seek information and the final action format;
it does not contain tool-call envelopes or a second copy of the tool catalog.

`ModelReply` reads native `function_call` output items. The decision loop dispatches
their validated arguments to the named plugin and returns `function_call_output`
with the same provider-issued `call_id`. The original model output is preserved in
that exchange. Text that resembles a tool call or logging event cannot invoke a tool.
Strict JSON Schema constrains final actions as well as tool arguments; nonblank strings
and two-element points are checked locally because DeepSeek does not support length
constraints in strict schemas.

`PluginResult.data` holds structured values for dependencies and trajectory logs.
`context_blocks` holds the plugin's model-facing explanation and images. The controller
uses those blocks in its current context and native tool results, without serializing
logging events, internal result versions, or bookkeeping into the prompt. Each model
request is saved separately in full for inspection, along with the provider response.

## Host responsibilities

The host supplies plugin_context snapshots, render_template, JSON Schema validation,
model/helper transport, image annotations, and robot access for conditional hooks.
Only named template placeholders are substituted, so literal JSON examples survive.
Information-tool contexts expose read-only observations/state, not robot actuation.
The planning-agent adapter can inspect the supplied history path. Helper failures,
including unlocalizable points, raise instead of fabricating valid structured output.

Current-result slots are separate from the protocol transcript and trajectory history.
Retiring a result removes its active context/annotation, not the historical evidence.
Historical tool messages must not be labelled as current. Preserve the assistant's
actual tool-call message and correlate its response with the same call ID. Automatic
prerequisite results are controller context and trace events, never fabricated
assistant calls or unmatched native tool responses.

`ContextUpdater` manages current results, observations, and completed action history.
`PluginController` resolves dependencies and schedules automatic contributions;
`DecisionLoop` constructs each model request from that current state. Automatic
contributions do not produce plugin lifecycle log entries. Actual interventions are
recorded as executed actions, and their resulting observations are captured before
the next model call. A handled grasp failure is not released repeatedly.

## Verification walkthroughs

Check persistent reuse/replacement/failure and episode reset; decision and observation
expiry; dependency updates; chains and shared prerequisites; invalid graphs; schema
errors; budget exhaustion; and direct actions with zero tool calls. Add a synthetic
tool with a non-prompt argument schema and multiple prerequisites to verify that the
controller does not depend on the current example pool.

These checks establish orchestration behavior, not model judgment. Later evaluate
selective calling against always-call and never-call baselines using task success
and call count.
