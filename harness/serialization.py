"""Lossless JSON boundaries and immutable plugin data."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

JSON: TypeAlias = None | bool | int | float | str | list["JSON"] | dict[str, "JSON"]
Object: TypeAlias = dict[str, Any]  # Untrusted JSON / provider boundary only.


def jsonable(value: Any) -> JSON:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Not serializable: {type(value).__name__}")


def dumps(value: Any) -> str:
    return json.dumps(
        jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def parse_object(text: str) -> Object:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> Object:
        result: Object = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value
