import logging
import pathlib
import sys
import warnings

LOGGER = logging.getLogger(__name__)


def deprecated(message, *, action="always"):
    """
    Show a deprecation warning, optionally filtered.

    Args:
        message (str): Message to display with ``DeprecationWarning``.
        action (str): Filter controlling whether warning is ignored, displayed, or
            turned into an error. https://docs.python.org/3/library/warnings.html#the-warnings-filter
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action, DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)


def get_config():
    """
    Get key configuration info about dev environment: OS, python, spacy, and textacy.

    Returns:
        dict
    """
    from spacy.about import __version__ as spacy_version
    from spacy.util import get_data_path
    from .about import __version__ as textacy_version

    return {
        "platform": sys.platform,
        "python": sys.version,
        "spacy": spacy_version,
        "spacy_models": [
            d.parts[-1]
            for d in get_data_path().iterdir()
            if (d.is_dir() or d.is_symlink())
            and d.parts[-1] not in {"__cache__", "__pycache__"}
        ],
        "textacy": textacy_version,
    }


def print_markdown(items):
    """
    Print ``items`` as a markdown-formatted list.
    Specifically useful when submitting config info on GitHub issues.

    Args:
        items (dict or Sequence[tuple])
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


def is_record(obj):
    """Check whether ``obj`` is a "record" -- that is, a (text, metadata) 2-tuple."""
    if (
        isinstance(obj, (tuple, list))
        and len(obj) == 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], dict)
    ):
        return True
    else:
        return False


def to_collection(val, val_type, col_type):
    """
    Validate and cast a value or values to a collection.

    Args:
        val (object): Value or values to validate and cast.
        val_type (type): Type of each value in collection, e.g. ``int`` or ``str``.
        col_type (type): Type of collection to return, e.g. ``tuple`` or ``set``.

    Returns:
        object: Collection of type ``col_type`` with values all of type ``val_type``.

    Raises:
        TypeError
    """
    if val is None:
        return None
    if isinstance(val, val_type):
        return col_type([val])
    elif isinstance(val, (tuple, list, set, frozenset)):
        if not all(isinstance(v, val_type) for v in val):
            raise TypeError("not all values are of type {}".format(val_type))
        return col_type(val)
    else:
        raise TypeError(
            "values must be {} or a collection thereof, not {}".format(
                val_type, type(val),
            )
        )


def to_bytes(s, *, encoding="utf-8", errors="strict"):
    """Coerce ``s`` to bytes.

    Args:
        s (str or bytes)
        encoding (str)
        errors (str)

    Returns:
        bytes
    """
    if isinstance(s, str):
        return s.encode(encoding, errors)
    elif isinstance(s, bytes):
        return s
    else:
        raise TypeError("`s` must be {}, not {}".format((str, bytes), type(s)))


def to_unicode(s, *, encoding="utf-8", errors="strict"):
    """
    Coerce ``s`` to unicode.

    Args:
        s (str or bytes)
        encoding (str)
        errors (str)

    Returns:
        str
    """
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, str):
        return s
    else:
        raise TypeError("`s` must be {}, not {}".format((str, bytes), type(s)))


def to_path(path):
    """
    Coerce ``path`` to a ``pathlib.Path``.

    Args:
        path (str or :class:`pathlib.Path`)

    Returns:
        :class:`pathlib.Path`
    """
    if isinstance(path, str):
        return pathlib.Path(path)
    elif isinstance(path, pathlib.Path):
        return path
    else:
        raise TypeError(
            "`path` must be {}, not {}".format((str, pathlib.Path), type(path))
        )


def validate_set_members(vals, val_type, valid_vals=None):
    """
    Validate values that must be of a certain type and (optionally) found among
    a set of known valid values.

    Args:
        vals (obj or Set[obj]): Value or values to validate.
        val_type (type or Tuple[type]): Type(s) of which all ``vals`` must be instances.
        valid_vals (Set[obj]): Set of valid values in which all ``vals`` must be found.

    Return:
        Set[obj]: Validated values.

    Raises:
        TypeError
        ValueError
    """
    vals = to_collection(vals, val_type, set)
    if valid_vals is not None:
        if not isinstance(valid_vals, set):
            valid_vals = set(valid_vals)
        if not all(val in valid_vals for val in vals):
            raise ValueError(
                "values {} are invalid; valid values are {}".format(
                    vals.difference(valid_vals), valid_vals,
                )
            )
    return vals


def validate_and_clip_range(range_vals, full_range, val_type=None):
    """
    Validate and clip range values.

    Args:
        range_vals (list or tuple): Range values, i.e. [start_val, end_val), to validate
            and, if necessary, clip. If None, the value is set to the corresponding
            value in ``full_range``.
        full_range (list or tuple): Full range of values, i.e. [min_val, max_val),
            within which ``range_vals`` must lie.
        val_type (type or Tuple[type]): Type(s) of which all ``range_vals``
            must be instances (unless val is None).

    Returns:
        tuple: Range for which null or too-small/large values have been
        clipped to the min/max valid values.

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
                "range must have 2 items -- (start, end) -- not {}".format(len(range_))
            )
    if val_type:
        for range_ in (range_vals, full_range):
            for val in range_:
                if val is not None and not isinstance(val, val_type):
                    raise TypeError(
                        "range value={} must be of type {}, not {}".format(
                            val, val_type, type(val)
                        )
                    )
    if range_vals[0] is None:
        range_vals = (full_range[0], range_vals[1])
    elif range_vals[0] < full_range[0]:
        LOGGER.info(
            "start of range %s < minimum valid value %s; clipping...",
            range_vals[0], full_range[0],
        )
        range_vals = (full_range[0], range_vals[1])
    if range_vals[1] is None:
        range_vals = (range_vals[0], full_range[1])
    elif range_vals[1] > full_range[1]:
        LOGGER.info(
            "end of range %s > maximum valid value %s; clipping...",
            range_vals[1], full_range[1],
        )
        range_vals = (range_vals[0], full_range[1])
    return tuple(range_vals)
