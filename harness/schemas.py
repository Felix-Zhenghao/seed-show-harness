"""Strict-compatible structural schemas; semantic constraints stay in Python."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from harness.serialization import Object, jsonable


def object_schema(properties: Object) -> Object:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


EMPTY_SCHEMA = object_schema({})
PROMPT_SCHEMA = object_schema({"prompt": {"type": "string"}})


def check_schema(schema: Object) -> None:
    """Reject unsupported declarations before spending a model request."""
    Draft202012Validator.check_schema(schema)
    if schema.get("type") != "object":
        raise ValueError("Strict schema root must be an object")
    allowed = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "enum",
        "items",
        "anyOf",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "format",
        "const",
        "title",
    }

    def visit(node: Object) -> None:
        unknown = set(node) - allowed
        if unknown:
            raise ValueError(f"Unsupported strict schema keywords: {sorted(unknown)}")
        if node.get("type") == "object":
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False or set(node.get("required", ())) != set(props):
                raise ValueError("Strict objects require every property and forbid extra properties")
            for child in props.values():
                visit(child)
        if "items" in node:
            visit(node["items"])
        for child in node.get("anyOf", []):
            visit(child)

    visit(schema)


def validate(schema: Object, value: Any) -> None:
    try:
        Draft202012Validator(schema).validate(jsonable(value))
    except ValidationError as error:
        raise ValueError(error.message) from error


def require_text(data: Mapping[str, Any], *names: str) -> None:
    for name in names:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonblank string")


def validate_prompt(data: Mapping[str, Any]) -> None:
    require_text(data, "prompt")
