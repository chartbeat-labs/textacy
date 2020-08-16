from typing import Any, Collection, Type


def value_invalid_msg(
    name: str,
    value: Any,
    valid_values: Collection[Any],
) -> str:
    return f"`{name}` value = {value} is invalid; value must be one of {valid_values}."


def type_invalid_msg(
    name: str,
    val_type: Type,
    valid_val_type: Type,
) -> str:
    return (
        f"`{name}` type = {val_type} is invalid; type must match {valid_val_type}."
    )
