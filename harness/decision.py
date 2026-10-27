"""A decision gathers information until the model emits one validated action."""

from __future__ import annotations

from harness.actions import DECISION_SCHEMA, Decision
from harness.context import ContextUpdater
from harness.models.deepseek import CompletionClient, IncompleteModelResponse
from harness.ports import EventSink
from harness.prompting import (
    controller_instructions,
    current_context,
    current_observation_input,
    decision_prefix,
    tool_result_output,
)
from harness.serialization import Object, parse_object
from plugins.contracts import tool_definitions
from plugins.controller import PluginController


class DecisionLoop:
    def __init__(
        self,
        client: CompletionClient,
        controller: PluginController,
        updater: ContextUpdater,
        events: EventSink,
        corrections: int = 2,
    ):
        self.client, self.controller, self.updater = client, controller, updater
        self.events, self.corrections = events, corrections

    def _correct_output(self, message: str, corrections: int, decision_id: int) -> int:
        self.events.record(
            "controller_output_rejected", decision_id=decision_id, error=message, correction=corrections
        )
        if corrections >= self.corrections:
            raise RuntimeError("Controller output correction limit exceeded")
        self.updater.error(f"Correct the rejected response: {message}")
        return corrections + 1

    def decide(self) -> Decision:
        exchange: list[Object] = []
        presented_result: tuple[str, int] | None = None
        retry: list[Object] = []
        corrections = 0
        while True:
            state = self.updater.state
            presented: frozenset[str] = frozenset()
            if presented_result is not None:
                name, version = presented_result
                record = state.results.get(name)
                if record is not None and record.version == version and self.controller.reusable(name):
                    presented = frozenset((name,))
                else:
                    exchange, presented_result = [], None
            # Rebuild the current state once. Keep the latest real tool exchange;
            # other active results are presented through their plugin-owned text.
            inputs = [
                *decision_prefix(state, self.controller.registry, presented),
                current_observation_input(state),
                current_context(state, self.controller.registry, presented),
                *exchange,
                *retry,
            ]
            try:
                reply = self.client.complete(
                    role="controller",
                    instructions=controller_instructions(state, self.controller.registry),
                    input_items=inputs,
                    output_schema=DECISION_SCHEMA,
                    tools=tool_definitions(self.controller.registry),
                    tool_choice="none" if state.calls_left == 0 else "auto",
                    decision_id=state.decision_id,
                )
            except IncompleteModelResponse as error:
                corrections = self._correct_output(str(error), corrections, state.decision_id)
                retry = [self._retry_instruction()]
                continue  # Never replay or execute partial tool calls.
            try:
                if not reply.tool_calls:
                    decision = Decision.parse(reply.text)
                    self.events.record(
                        "action_proposed",
                        decision_id=state.decision_id,
                        action=decision.action.value,
                        rationale=decision.rationale,
                    )
                    return decision
                if len(reply.tool_calls) != 1:
                    raise ValueError("Call one information tool at a time and wait for its result")
                if state.calls_left == 0:
                    raise ValueError("Choose an action using the information already available")
            except (ValueError, TypeError) as error:
                corrections = self._correct_output(str(error), corrections, state.decision_id)
                retry = [
                    *reply.with_tool_results(
                        {
                            call.call_id: f"This tool call was not executed. {error}"
                            for call in reply.tool_calls
                        }
                    ),
                    self._retry_instruction(),
                ]
                continue
            call = reply.tool_calls[0]
            # A native call takes precedence over accompanying assistant text.
            # Only a later response without calls can select an executable action.
            arguments: Object | str
            try:
                # Parsing failures are invocation failures and consume a slot too.
                try:
                    arguments = parse_object(call.arguments)
                except ValueError:
                    arguments = call.arguments
                result = self.controller.invoke(call.name, arguments)
                output = tool_result_output(result)
                presented_result = (call.name, self.updater.state.results[call.name].version)
            except Exception as error:
                self.updater.error(f"{call.name}: {error}")
                output = f"The {call.name} request failed: {error}"
                presented_result = None
            exchange = reply.with_tool_results({call.call_id: output})
            retry = []

    def _retry_instruction(self) -> Object:
        instruction = (
            "The last response was not executed. Return one final JSON object with exactly "
            "rationale and action_choice."
        )
        if self.updater.state.calls_left != 0:
            instruction += (
                " If you need information, call one of the available tools using its declared arguments "
                "and wait for the result before choosing an action."
            )
        return {"role": "user", "content": [{"type": "input_text", "text": instruction}]}
