import sys
import warnings


def deprecated(message, action="always"):
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


def to_bytes(s, encoding="utf-8", errors="strict"):
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


def to_unicode(s, encoding="utf-8", errors="strict"):
    """Coerce ``s`` to unicode.

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
