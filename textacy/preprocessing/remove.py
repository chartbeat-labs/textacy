# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import unicodedata

from .resources import _get_punct_translation_table


def remove_accents(text, method="unicode"):
    """
    Remove accents from any accented unicode characters in ``text``, either by
    replacing them with ASCII equivalents or removing them entirely.

    Args:
        text (str)
        method ({"unicode", "ascii"}): If "unicode", remove accents from any
            unicode symbol with a direct ASCII equivalent; if "ascii",
            remove accented char for any unicode symbol.

            .. note:: The "ascii" method is notably faster than "unicode", but less good.

    Returns:
        str

    Raises:
        ValueError: If ``method`` is not in {"unicode", "ascii"}.
    """
    if method == "unicode":
        return "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    elif method == "ascii":
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )
    else:
        raise ValueError(
            "method = '{}' is invalid; value must be in {}".format(
                method, {"unicode", "ascii"})
        )


def remove_punctuation(text, marks=None):
    """
    Remove punctuation from ``text`` by replacing all instances of ``marks``
    with whitespace.

    Args:
        text (str)
        marks (str): Remove only those punctuation marks specified here.
            For example, ",;:" removes commas, semi-colons, and colons.
            If None, *all* unicode punctuation marks are removed.

    Returns:
        str

    Note:
        When ``marks=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise, a regular expression is used.
        The former's performance is about 5-10x faster.
    """
    if marks:
        return re.sub("[{}]+".format(re.escape(marks)), " ", text, flags=re.UNICODE)
    else:
        return text.translate(_get_punct_translation_table())
