"""
Token-based Metrics
-------------------

:mod:`textacy.similarity.edits`: Normalized similarity metrics built on token-based
algorithms that identify and count similar tokens between one sequence and another,
and don't rely on the *ordering* of those tokens.
"""
import collections
import math
from typing import Iterable


def jaccard(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings as sets
    using the Jaccard index.

    Args:
        seq1
        seq2

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences of strings

    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    set1 = set(seq1)
    set2 = set(seq2)
    try:
        return len(set1 & set2) / len(set1 | set2)
    except ZeroDivisionError:
        return 0.0


def sorensen_dice(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings as sets
    using the Sørensen-Dice index, which is similar to the Jaccard index.

    Args:
        seq1
        seq2

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences

    Reference:
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """
    set1 = set(seq1)
    set2 = set(seq2)
    try:
        return 2 * len(set1 & set2) / (len(set1) + len(set2))
    except ZeroDivisionError:
        return 0.0


def tversky(
    seq1: Iterable[str], seq2: Iterable[str], *, alpha: float = 1.0, beta: float = 1.0
) -> float:
    """
    Measure the similarity between two sequences of strings as sets
    using the (symmetric) Tversky index, which is a generalization of
    Jaccard (``alpha=0.5, beta=2.0``) and Sørensen-Dice (``alpha=0.5, beta=1.0``).

    Args:
        seq1
        seq2
        alpha
        beta

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences

    Reference:
        https://en.wikipedia.org/wiki/Tversky_index
    """
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = len(set1 & set2)
    set1_not_set2 = len(set1 - set2)
    set2_not_set1 = len(set2 - set1)
    a = min(set1_not_set2, set2_not_set1)
    b = max(set1_not_set2, set2_not_set1)
    try:
        return intersection / (intersection + (beta * (alpha * a + (1 - alpha) * b)))
    except ZeroDivisionError:
        return 0.0


def cosine(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings as sets
    using the Otsuka-Ochiai variation of cosine similarity (which is equivalent
    to the usual formulation when values are binary).

    Args:
        seq1
        seq2

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences

    Reference:
        https://en.wikipedia.org/wiki/Cosine_similarity#Otsuka-Ochiai_coefficient
    """
    set1 = set(seq1)
    set2 = set(seq2)
    try:
        return len(set1 & set2) / math.sqrt(len(set1) * len(set2))
    except ZeroDivisionError:
        return 0.0


def bag(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings (*not* as sets)
    using the "bag distance" measure, which can be considered an approximation
    of edit distance.

    Args:
        seq1
        seq2

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences

    Reference:
        Bartolini, Ilaria, Paolo Ciaccia, and Marco Patella. "String matching with
        metric trees using an approximate distance." International Symposium on String
        Processing and Information Retrieval. Springer, Berlin, Heidelberg, 2002.
    """
    bag1 = collections.Counter(seq1)
    bag2 = collections.Counter(seq2)
    max_diff = max(sum((bag1 - bag2).values()), sum((bag2 - bag1).values()))
    max_len = max(sum(bag1.values()), sum(bag2.values()))
    try:
        return 1.0 - (max_diff / max_len)
    except ZeroDivisionError:
        return 0.0
