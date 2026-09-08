from __future__ import annotations

import math
import types
from dataclasses import fields, is_dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


def _type_name(annotation: Any) -> str:
    if annotation is type(None):
        return "null"
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _validate_value(value: Any, annotation: Any, path: str) -> None:
    if annotation is Any:
        return

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {Union, types.UnionType}:
        failures: list[ValueError] = []
        for candidate in arguments:
            try:
                _validate_value(value, candidate, path)
                return
            except ValueError as error:
                failures.append(error)
        expected = " | ".join(_type_name(candidate) for candidate in arguments)
        raise ValueError(f"{path} must have type {expected}; got {type(value).__name__}")

    if origin is Literal:
        if not any(
            type(value) is type(candidate) and value == candidate for candidate in arguments
        ):
            choices = ", ".join(repr(candidate) for candidate in arguments)
            raise ValueError(f"{path} must be one of {choices}; got {value!r}")
        return

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{path} must have type list; got {type(value).__name__}")
        item_type = arguments[0] if arguments else Any
        for index, item in enumerate(value):
            _validate_value(item, item_type, f"{path}[{index}]")
        return

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must have type tuple; got {type(value).__name__}")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            for index, item in enumerate(value):
                _validate_value(item, arguments[0], f"{path}[{index}]")
        elif len(value) != len(arguments):
            raise ValueError(f"{path} must contain {len(arguments)} items; got {len(value)}")
        else:
            for index, (item, item_type) in enumerate(zip(value, arguments, strict=True)):
                _validate_value(item, item_type, f"{path}[{index}]")
        return

    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a JSON object; got {type(value).__name__}")
        validate_dataclass_payload(annotation, value, path=path)
        return

    # bool is an int subclass in Python, so all numeric checks must exclude it.
    if annotation is bool:
        valid = type(value) is bool
    elif annotation is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif annotation is str:
        valid = isinstance(value, str)
    elif annotation is type(None):
        valid = value is None
    else:
        valid = isinstance(value, annotation)
    if not valid:
        raise ValueError(
            f"{path} must have type {_type_name(annotation)}; got {type(value).__name__}"
        )
    if annotation is float and not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite; got {value!r}")


def validate_dataclass_payload(
    cls: type[Any], values: dict[str, Any], *, path: str | None = None
) -> None:
    """Validate JSON-derived values against a dataclass without coercing truthy strings."""

    if not isinstance(values, dict):
        raise ValueError(f"{path or cls.__name__} must be a JSON object")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    prefix = path or cls.__name__
    for name, value in values.items():
        _validate_value(value, hints[name], f"{prefix}.{name}")


__all__ = ["validate_dataclass_payload"]
