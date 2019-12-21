"""
Normalize
---------

Normalize aspects of raw text that may vary in problematic ways.
"""
import re
import unicodedata

from .resources import (
    QUOTE_TRANSLATION_TABLE,
    RE_HYPHENATED_WORD,
    RE_LINEBREAK,
    RE_NONBREAKING_SPACE,
    RE_ZWSP,
)


def normalize_hyphenated_words(text):
    """
    Normalize words in ``text`` that have been split across lines by a hyphen
    for visual consistency (aka hyphenated) by joining the pieces back together,
    sans hyphen and whitespace.

    Args:
        text (str)

    Returns:
        str
    """
    return RE_HYPHENATED_WORD.sub(r"\1\2", text)


def normalize_quotation_marks(text):
    """
    Normalize all "fancy" single- and double-quotation marks in ``text``
    to just the basic ASCII equivalents. Note that this will also normalize fancy
    apostrophes, which are typically represented as single quotation marks.

    Args:
        text (str)

    Returns:
        str
    """
    return text.translate(QUOTE_TRANSLATION_TABLE)


def normalize_repeating_chars(text, *, chars, maxn=1):
    """
    Normalize repeating characters in ``text`` by truncating their number of consecutive
    repetitions to ``maxn``.

    Args:
        text (str)
        chars (str): One or more characters whose consecutive repetitions are to be
            normalized, e.g. "." or "?!".
        maxn (int): Maximum number of consecutive repetitions of ``chars`` to which
            longer repetitions will be truncated.

    Returns:
        str
    """
    return re.sub(r"({}){{{},}}".format(re.escape(chars), maxn + 1), chars * maxn, text)


def normalize_unicode(text, *, form="NFC"):
    """
    Normalize unicode characters in ``text`` into canonical forms.

    Args:
        text (str)
        form ({"NFC", "NFD", "NFKC", "NFKD"}): Form of normalization applied to
            unicode characters. For example, an "e" with accute accent "´" can be
            written as "e´" (canonical decomposition, "NFD") or "é" (canonical
            composition, "NFC"). Unicode can be normalized to NFC form
            without any change in meaning, so it's usually a safe bet. If "NFKC",
            additional normalizations are applied that can change characters' meanings,
            e.g. ellipsis characters are replaced with three periods.

    See Also:
        https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text):
    """
    Replace all contiguous zero-width spaces with an empty string, line-breaking spaces
    with a single newline, and non-breaking spaces with a single space, then
    strip any leading/trailing whitespace.

    Args:
        text (str)

    Returns:
        str
    """
    text = RE_ZWSP.sub("", text)
    text = RE_LINEBREAK.sub(r"\n", text)
    text = RE_NONBREAKING_SPACE.sub(" ", text)
    return text.strip()
