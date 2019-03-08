from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import warnings

from . import compat


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
            compat.unicode_(k).replace("\n", " "), compat.unicode_(v).replace("\n", " ")
        )
        for k, v in items
    )
    print("{}".format("\n".join(md_items)))
