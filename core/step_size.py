"""Validate and resolve model step sizes for a single semantic decision."""
from __future__ import annotations

import logging
import math

from core.action_units import MOVE_ATOMS, RT_ATOMS


SIZE_TOKENS = {"distance_m": MOVE_ATOMS, "angle_deg": RT_ATOMS}
# Match the existing configuration limits; operators may tighten the override bounds.
SIZE_LIMITS = {"distance_m": (0.0001, 1.0), "angle_deg": (0.01, 180.0)}
logger = logging.getLogger(__name__)


def size_bounds(actions: dict, field: str) -> tuple[float, float]:
    low, high = SIZE_LIMITS[field]
    return actions.get(f"min_{field}", low), actions.get(f"max_{field}", high)


def has_explicit_size(answer: dict) -> bool:
    """Null is a request for the default, not an explicit size."""
    return any(answer.get(field) is not None for field in SIZE_TOKENS)


def parse_step_size(token: str, answer: dict, actions: dict) -> dict:
    """Keep original requests for diagnostics; only expose validated magnitudes."""
    result = {
        "requested": {field: answer[field] for field in SIZE_TOKENS if field in answer},
        "distance_m": None,
        "angle_deg": None,
        "diagnostics": [],
    }
    for field, tokens in SIZE_TOKENS.items():
        value = answer.get(field)
        if value is None:
            continue
        low, high = size_bounds(actions, field)
        if token not in tokens:
            error = f"not applicable to {token}"
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            error = "must be a finite, positive number"
        elif not 0 < value <= high or not math.isfinite(value):
            error = f"must be finite and positive within [{low:g}, {high:g}]"
        elif value < low:
            error = f"outside configured bounds [{low:g}, {high:g}]"
        else:
            result[field] = float(value)
            continue
        diagnostic = f"Ignoring {field}={value!r} for {token}: {error}; using default"
        result["diagnostics"].append(diagnostic)
        logger.warning(diagnostic)
    return result


def resolve_step_size(token: str, override: dict | None, *, distance_m: float,
                      angle_deg: float) -> dict:
    """Use the current defaults for every field without a valid applicable override."""
    override = override or {}
    field = "distance_m" if token in MOVE_ATOMS else "angle_deg" if token in RT_ATOMS else None
    value = override.get(field) if field else None
    return {
        "requested": dict(override.get("requested", {})),
        "effective_distance_m": (
            value if value is not None else distance_m
        ) if token in MOVE_ATOMS else None,
        "effective_angle_deg": (
            value if value is not None else angle_deg
        ) if token in RT_ATOMS else None,
        "source": "model" if value is not None else "default",
        "diagnostics": list(override.get("diagnostics", [])),
    }
