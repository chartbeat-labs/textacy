"""
Hybrid Metrics
--------------

:mod:`textacy.similarity.hybrid`: Normalized similarity metrics that combine edit-,
token-, and/or sequence-based algorithms.
"""
from __future__ import annotations

from typing import Callable, Sequence

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

    See Also:
        :func:`textacy.similarity.edits.levenshtein()`
    """
    str1 = _to_prepared_str(s1)
    str2 = _to_prepared_str(s2)
    return edits.levenshtein(str1, str2)


def _to_prepared_str(s: str | Sequence[str]) -> str:
    """
    Remove all characters from ``s`` except letters and numbers, strip whitespace,
    and force everything to lower-case; then sort tokens before re-joining into
    a single string.
    """
    tokens = (
        constants.RE_ALNUM.findall(s.lower())
        if isinstance(s, str)
        else [tok.lower().strip() for tok in s]
    )
    return " ".join(sorted(tokens))


def monge_elkan(
    seq1: Sequence[str],
    seq2: Sequence[str],
    sim_func: Callable[[str, str], float] = edits.levenshtein,
) -> float:
    """
    Measure the similarity between two sequences of strings using the (symmetric)
    Monge-Elkan method, which takes the average of the maximum pairwise similarity
    between the tokens in each sequence as compared to those in the other sequence.

    Args:
        seq1
        seq2
        sim_func: Callable that computes a string-to-string similarity metric;
            by default, Levenshtein edit distance.

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings.

    See Also:
        :func:`textacy.similarity.edits.levenshtein()`
    """
    if not seq1 or not seq2:
        return 0.0

    sum_maxsim1 = sum(
        max(sim_func(tok1, tok2) for tok2 in seq2)
        for tok1 in seq1
    )
    sum_maxsim2 = sum(
        max(sim_func(tok2, tok1) for tok1 in seq1)
        for tok2 in seq2
    )
    return ((sum_maxsim1 / len(seq1)) + (sum_maxsim2 / len(seq2))) / 2
