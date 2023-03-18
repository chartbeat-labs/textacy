"""
Remove
------

:mod:`textacy.preprocessing.remove`: Remove aspects of raw text that may be unwanted
for certain use cases.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Collection, Optional

from .. import utils
from . import resources


def accents(text: str, *, fast: bool = False) -> str:
    """
    Remove accents from any accented unicode characters in ``text``, either by
    replacing them with ASCII equivalents or removing them entirely.

    Args:
        text
        fast: If False, accents are removed from any unicode symbol
            with a direct ASCII equivalent; if True, accented chars
            for all unicode symbols are removed, regardless.

            .. note:: ``fast=True`` can be significantly faster than ``fast=False``,
               but its transformation of ``text`` is less "safe" and more likely
               to result in changes of meaning, spelling errors, etc.

    Returns:
        str

    See Also:
        For a more powerful (but slower) alternative, check out ``unidecode``:
        https://github.com/avian2/unidecode
    """
    if fast is False:
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    else:
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )


def brackets(
    text: str,
    *,
    only: Optional[str | Collection[str]] = None,
) -> str:
    """
    Remove text within curly {}, square [], and/or round () brackets, as well as
    the brackets themselves.

    Args:
        text
        only: Remove only those bracketed contents as specified here: "curly", "square",
            and/or "round". For example, ``"square"`` removes only those contents found
            between square brackets, while ``["round", "square"]`` removes those contents
            found between square or round brackets, but not curly.

    Returns:
        str

    Note:
        This function relies on regular expressions, applied sequentially for curly,
        square, then round brackets; as such, it doesn't handle nested brackets of the
        same type and may behave unexpectedly on text with "wild" use of brackets.
        It should be fine removing structured bracketed contents, as is often used,
        for instance, to denote in-text citations.
    """
    only = utils.to_set(only) if only is not None else None
    if only is None or "curly" in only:
        text = resources.RE_BRACKETS_CURLY.sub("", text)
    if only is None or "square" in only:
        text = resources.RE_BRACKETS_SQUARE.sub("", text)
    if only is None or "round" in only:
        text = resources.RE_BRACKETS_ROUND.sub("", text)
    return text


def html_tags(text: str) -> str:
    """
    Remove HTML tags from ``text``, returning just the text found between tags
    and other non-data elements.

    Args:
        text

    Returns:
        str

    Note:
        This function relies on the stdlib :class:`html.parser.HTMLParser` and
        doesn't do anything fancy. For a better and potentially faster solution,
        consider using ``lxml`` and/or ``beautifulsoup4``.
    """
    parser = resources.HTMLTextExtractor()
    parser.feed(text)
    return parser.get_text()


def punctuation(
    text: str,
    *,
    only: Optional[str | Collection[str]] = None,
) -> str:
    """
    Remove punctuation from ``text`` by replacing all instances of punctuation
    (or a subset thereof specified by ``only``) with whitespace.

    Args:
        text
        only: Remove only those punctuation marks specified here. For example,
            ``"."`` removes only periods, while ``[",", ";", ":"]`` removes commas,
            semicolons, and colons; if None, all unicode punctuation marks are removed.

    Returns:
        str

    Note:
        When ``only=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise, a regular expression is used.
        The former's performance can be up to an order of magnitude faster.
    """
    only = utils.to_set(only) if only is not None else None
    if only is not None:
        return re.sub("[{}]+".format(re.escape("".join(only))), " ", text)
    else:
        return text.translate(resources.PUNCT_TRANSLATION_TABLE)
