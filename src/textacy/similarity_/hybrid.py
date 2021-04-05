from __future__ import annotations

from typing import Sequence

from . import edits
from .. import constants


def token_sort_ratio(s1: str | Sequence[str], s2: str | Sequence[str]) -> float:
    """
    Measure the similarity between two strings or sequences of strings
    using Levenshtein distance, only with non-alphanumeric characters removed
    and the ordering of tokens in each sorted before comparison.

    Args:
        s1
        s2

    Returns:
        Similarity between ``s1`` and ``s2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings.
    """
    str1 = _process_and_sort(s1)
    str2 = _process_and_sort(s2)
    return edits.levenshtein(str1, str2)


def _process_and_sort(s: str | Sequence[str]) -> str:
    """
    Remove all characters from ``s`` except letters and numbers, strip whitespace,
    and force everything to lower-case; then sort tokens before re-joining into
    a single string.
    """
    tokens = (
        constants.RE_ALNUM.findall(s.lower())
        if isinstance(s, str)
        else [tok.lower() for tok in s]
    )
    return " ".join(sorted(tokens))
