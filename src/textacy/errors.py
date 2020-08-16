from typing import Any, Collection


def value_not_valid(name: str, value: Any, valid_values: Collection[Any]):
    raise ValueError(f"{name} = {value} is invalid; must be one of {valid_values}.")
