"""Compose reusable instructions before live state and observation images."""
from plugins.action_chunk import build as action_chunk
from plugins.affordance import build as affordance
from plugins.auto_release import build as auto_release
from plugins.base import build as base
from plugins.decision_json import build as decision_json
from plugins.decision_output import build as decision_output
from plugins.decision_token import build as decision_token
from plugins.extra_context import build as extra_context
from plugins.mem_text import build as mem_text
from plugins.observations import build as observations
from plugins.proprioception import build as proprioception
from plugins.recovery import build as recovery
from plugins.strict_token_retry import build as strict_token_retry
from plugins.subgoal import build as subgoal
from plugins.variable_step import build as variable_step
from plugins.wrist_visibility import build as wrist_visibility

from .content import join_content


def compose(context, *builders):
    return join_content(builder(context) for builder in builders)


def controller_prompt(context):
    return compose(
        context, base.build_instructions, extra_context.build,
        wrist_visibility.build, action_chunk.build, variable_step.build_instructions,
        mem_text.build_instructions,
        base.build_state, variable_step.build_state, affordance.build, proprioception.build,
        mem_text.build_state, recovery.build, auto_release.build,
    )


def planner_prompt(context):
    return compose(context, base.build_instructions, extra_context.build, subgoal.build,
                   base.build_state)


def decision_prompt(context):
    return join_content((context["previous_prompt"], compose(context, decision_output.build, decision_json.build)))


def token_prompt(context):
    return decision_token.build(context)


def token_retry_prompt(context):
    return join_content((context["previous_prompt"], strict_token_retry.build(context)))


def model_message(context):
    # Prefix caches stop matching at the first changed token or image. Preserve
    # plugin-owned image positions, then append the changing camera observations.
    return join_content((context["previous_prompt"], observations.build(context)))
