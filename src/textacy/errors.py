"""
:mod:`textacy.errors`: Helper functions for making consistent errors.
"""
from typing import Any, Collection


def value_invalid_msg(name: str, value: Any, valid_values: Collection[Any]) -> str:
    return f"`{name}` value = {value} is invalid; value must be one of {valid_values}."


def type_invalid_msg(name: str, val_type, valid_val_type) -> str:
    return f"`{name}` type = {val_type} is invalid; type must match {valid_val_type}."
