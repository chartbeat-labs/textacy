"""
:mod:`textacy.utils`: Variety of general-purpose utility functions for inspecting /
validating / transforming args and facilitating meta package tasks.
"""
from __future__ import annotations

import inspect
import logging
import pathlib
import sys
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

from . import errors as errors_
from . import types


LOGGER = logging.getLogger(__name__)


_KW_PARAM_KINDS = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
}


def deprecated(
    message: str,
    *,
    action: Literal[
        "default", "error", "ignore", "always", "module", "once"
    ] = "always",
):
    """
    Show a deprecation warning, optionally filtered.

    Args:
        message: Message to display with ``DeprecationWarning``.
        action: Filter controlling whether warning is ignored, displayed, or turned
            into an error. For reference:

    See Also:
        https://docs.python.org/3/library/warnings.html#the-warnings-filter
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action, DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def get_config() -> dict[str, Any]:
    """
    Get key configuration info about dev environment: OS, python, spacy, and textacy.

    Returns:
        dict
    """
    from spacy.about import __version__ as spacy_version
    from spacy.util import get_installed_models

    from ._version import __version__ as textacy_version

    return {
        "platform": sys.platform,
        "python": sys.version,
        "spacy": spacy_version,
        "spacy_models": get_installed_models(),
        "textacy": textacy_version,
    }


def print_markdown(items: dict[Any, Any] | Iterable[tuple[Any, Any]]):
    """
    Print ``items`` as a markdown-formatted list.
    Specifically useful when submitting config info on GitHub issues.

    Args:
        items
    """
    if isinstance(items, dict):
        items = list(items.items())
    md_items = (
        "- **{}:** {}".format(
            to_unicode(k).replace("\n", " "), to_unicode(v).replace("\n", " ")
        )
        for k, v in items
    )
    print("{}".format("\n".join(md_items)))


def is_record(obj: Any) -> bool:
    """Check whether ``obj`` is a "record" -- that is, a (text, metadata) 2-tuple."""
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], dict)
    ):
        return True
    else:
        return False


def to_list(val: Any) -> list:
    """Cast ``val`` into a list, if necessary and possible."""
    if isinstance(val, list):
        return val
    elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return list(val)
    else:
        return [val]


def to_set(val: Any) -> set:
    """Cast ``val`` into a set, if necessary and possible."""
    if isinstance(val, set):
        return val
    elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return set(val)
    else:
        return {val}


def to_tuple(val: Any) -> tuple:
    """Cast ``val`` into a tuple, if necessary and possible."""
    if isinstance(val, tuple):
        return val
    elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
        return tuple(val)
    else:
        return (val,)


def to_collection(
    val: Optional[types.AnyVal | Collection[types.AnyVal]],
    val_type: Type[Any] | tuple[Type[Any], ...],
    col_type: Type[Any],
) -> Optional[Collection[types.AnyVal]]:
    """
    Validate and cast a value or values to a collection.

    Args:
        val (object): Value or values to validate and cast.
        val_type (type): Type of each value in collection, e.g. ``int`` or ``(str, bytes)``.
        col_type (type): Type of collection to return, e.g. ``tuple`` or ``set``.

    Returns:
        Collection of type ``col_type`` with values all of type ``val_type``.

    Raises:
        TypeError
    """
    if val is None:
        return None
    if isinstance(val, val_type):
        return col_type([val])
    elif isinstance(val, (tuple, list, set, frozenset)):
        if not all(isinstance(v, val_type) for v in val):
            raise TypeError(f"not all values are of type {val_type}")
        return col_type(val)
    else:
        # TODO: use standard error message, maybe?
        raise TypeError(
            f"values must be {val_type} or a collection thereof, not {type(val)}"
        )


def to_bytes(
    s: types.AnyStr, *, encoding: str = "utf-8", errors: str = "strict"
) -> bytes:
    """Coerce string ``s`` to bytes."""
    if isinstance(s, str):
        return s.encode(encoding, errors)
    elif isinstance(s, bytes):
        return s
    else:
        raise TypeError(errors_.type_invalid_msg("s", type(s), Union[str, bytes]))


def to_unicode(
    s: types.AnyStr, *, encoding: str = "utf-8", errors: str = "strict"
) -> str:
    """Coerce string ``s`` to unicode."""
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, str):
        return s
    else:
        raise TypeError(errors_.type_invalid_msg("s", type(s), Union[str, bytes]))


def to_path(path: types.PathLike) -> pathlib.Path:
    """
    Coerce ``path`` to a ``pathlib.Path``.

    Args:
        path

    Returns:
        :class:`pathlib.Path`
    """
    if isinstance(path, str):
        return pathlib.Path(path)
    elif isinstance(path, pathlib.Path):
        return path
    else:
        raise TypeError(
            errors_.type_invalid_msg("path", type(path), Union[str, pathlib.Path])
        )


def validate_set_members(
    vals: types.AnyVal | set[types.AnyVal],
    val_type: Type[Any] | tuple[Type[Any], ...],
    valid_vals: Optional[set[types.AnyVal]] = None,
) -> set[types.AnyVal]:
    """
    Validate values that must be of a certain type and (optionally) found among
    a set of known valid values.

    Args:
        vals: Value or values to validate.
        val_type: Type(s) of which all ``vals`` must be instances.
        valid_vals: Set of valid values in which all ``vals`` must be found.

    Return:
        set[obj]: Validated values.

    Raises:
        TypeError
        ValueError
    """
    vals = cast(set, to_collection(vals, val_type, set))
    if valid_vals is not None:
        if not isinstance(valid_vals, set):
            valid_vals = set(valid_vals)
        if not all(val in valid_vals for val in vals):
            raise ValueError(
                f"values {vals.difference(valid_vals)} are invalid; "
                f"valid values are {valid_vals}"
            )
    return vals


def validate_and_clip_range(
    range_vals: tuple[types.AnyVal, types.AnyVal],
    full_range: tuple[types.AnyVal, types.AnyVal],
    val_type: Optional[Type[Any] | tuple[Type[Any], ...]] = None,
) -> tuple[types.AnyVal, types.AnyVal]:
    """
    Validate and clip range values.

    Args:
        range_vals: Range values, i.e. [start_val, end_val), to validate
            and, if necessary, clip. If None, the value is set to the corresponding
            value in ``full_range``.
        full_range: Full range of values, i.e. [min_val, max_val),
            within which ``range_vals`` must lie.
        val_type: Type(s) of which all ``range_vals`` must be instances,
            unless val is None.

    Returns:
        Range for which null or too-small/large values have been clipped
        to the min/max valid values.

    Raises:
        TypeError
        ValueError
    """
    for range_ in (range_vals, full_range):
        if not isinstance(range_, (list, tuple)):
            raise TypeError(
                "range must be of type {}, not {}".format({list, tuple}, type(range_))
            )
        if len(range_) != 2:
            raise ValueError(
                f"range must have 2 items -- (start, end) -- not {len(range_)}"
            )
    if val_type:
        for range_ in (range_vals, full_range):
            for val in range_:
                if val is not None and not isinstance(val, val_type):
                    raise TypeError(
                        f"range value={val} must be of type {val_type}, not {type(val)}"
                    )
    if range_vals[0] is None:
        range_vals = (full_range[0], range_vals[1])
    elif range_vals[0] < full_range[0]:  # type: ignore
        LOGGER.info(
            "start of range %s < minimum valid value %s; clipping...",
            range_vals[0],
            full_range[0],
        )
        range_vals = (full_range[0], range_vals[1])
    if range_vals[1] is None:
        range_vals = (range_vals[0], full_range[1])
    elif range_vals[1] > full_range[1]:  # type: ignore
        LOGGER.info(
            "end of range %s > maximum valid value %s; clipping...",
            range_vals[1],
            full_range[1],
        )
        range_vals = (range_vals[0], full_range[1])
    return cast(tuple[Any, Any], tuple(range_vals))


def get_kwargs_for_func(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Get the set of keyword arguments from ``kwargs`` that are used by ``func``.
    Useful when calling a func from another func and inferring its signature
    from provided ``**kwargs``.
    """
    if not kwargs:
        return {}

    func_params = {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind in _KW_PARAM_KINDS
    }
    func_kwargs = {
        kwarg: value for kwarg, value in kwargs.items() if kwarg in func_params
    }
    return func_kwargs


def text_to_char_ngrams(text: str, n: int, *, pad: bool = False) -> tuple[str, ...]:
    """
    Convert a text string into an ordered sequence of character ngrams.

    Args:
        text
        n: Number of characters to concatenate in each ``n``-gram.
        pad: If True, pad ``text`` by adding ``n - 1`` "_" characters on either side;
            if False, leave ``text`` as-is.

    Returns:
        Ordered sequence of character ngrams.
    """
    if pad is True and n > 1:
        pad_chars = "_" * (n - 1)
        text = f"{pad_chars}{text}{pad_chars}"
    return tuple(text[i : i + n] for i in range(len(text) - n + 1))


def get_function_names(module, ignore_private: bool = True) -> Iterable[str]:
    """
    Get names of functions in ``module``, optionally ignoring private members.

    Args:
        module
        ignore_private

    Returns:
        Alphabetically ordered sequence of function names.
    """
    names = (name for name, _ in inspect.getmembers(module, inspect.isfunction))
    if ignore_private is True:
        names = (name for name in names if not name.startswith("_"))
    yield from names
