# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import unicodedata

from .resources import RE_HYPHENATED_WORD, RE_LINEBREAK, RE_NONBREAKING_SPACE


def normalize_contractions(text):
    """
    Replace *English* contractions in ``text`` with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.

    TODO: Make this better, and multi-lingual.
    """
    # standard
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
        r"\1\2 not",
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
        r"\1\2 will",
        text,
    )
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
        r"\1\2 have",
        text,
    )
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text


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


def normalize_unicode(text, form="NFC"):
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
    Replace all contiguous line-breaking whitespaces with a single newline and
    all contiguous non-breaking whitespaces with a single space, then
    strip any leading/trailing whitespace.

    Args:
        text (str)

    Returns:
        str
    """
    return RE_NONBREAKING_SPACE.sub(" ", RE_LINEBREAK.sub(r"\n", text)).strip()
