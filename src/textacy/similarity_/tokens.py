"""
Token-based Similarity
----------------------

:mod:`textacy.similarity_.edits`: Normalized similarity metrics built on token-based
algorithms that identify and count similar tokens between one sequence and another.
"""
import collections
import math
from typing import Iterable

from . import edits


def jaccard(
    seq1: Iterable[str],
    seq2: Iterable[str],
    fuzzy_match: bool = False,
    match_threshold: float = 0.8,
) -> float:
    """
    Measure the similarity between two sequences of strings using Jaccard metric,
    with optional fuzzy matching of not-identical pairs.

    Args:
        seq1
        seq2
        fuzzy_match: If True, allow for fuzzy matching in addition to the
            usual identical matching of pairs between input vectors
        match_threshold: Value in the interval [0.0, 1.0]; fuzzy comparisons
            with a score >= this value will be considered matches

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings or sequences of strings

    Raises:
        ValueError: if ``match_threshold`` is not a valid float

    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    # TODO: maybe get rid of the fuzzy matching stuff?
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError(
            f"match_threshold={match_threshold} is invalid; "
            "it must be a float in the interval [0.0, 1.0]"
        )
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if fuzzy_match is True:
        for item1 in set1.difference(set2):
            if (
                max(edits.token_sort_ratio(item1, item2) for item2 in set2) >=
                match_threshold
            ):
                intersection += 1
        for item2 in set2.difference(set1):
            if (
                max(edits.token_sort_ratio(item2, item1) for item1 in set1) >=
                match_threshold
            ):
                intersection += 1
    elif fuzzy_match is True:
        raise ValueError("fuzzy matching not possible with str inputs")

    return intersection / union


def sorensen_dice(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings using Sørensen-Dice index,
    which TODO

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
    return 2 * len(set1 & set2) / (len(set1) + len(set2))


def tversky(
    seq1: Iterable[str], seq2: Iterable[str], alpha: float = 1.0, beta: float = 1.0
) -> float:
    """
    Measure the similarity between two sequences of strings using the (symmetric)
    Tversky index, which TODO

    Args:
        seq1
        seq2
        alpha
        beta

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences

    Note:
        This metric is a generalization of Jaccard (``alpha=0.5``, ``beta=2.0``) and
        Sørensen-Dice (``alpha=0.5``, ``beta=1.0``).

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
    return intersection / (intersection + (beta * ((alpha * a) + ((1 - alpha) * b))))


def cosine(seq1: Iterable[str], seq2: Iterable[str]) -> float:
    """
    Measure the similarity between two sequences of strings using the Otsuka-Ochiai
    variation of cosine similarity (which is equivalent to the usual formulation when
    values are binary).

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
    Measure the similarity between two sequences of strings using the TODO

    Args:
        seq1
        seq2

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences
    """
    bag1 = collections.Counter(seq1)
    bag2 = collections.Counter(seq2)
    max_set_diff = max(sum((bag1 - bag2).values()), sum((bag2 - bag1).values()))
    max_len = max(sum(bag1.values()), sum(bag2.values()))
    try:
        return 1.0 - (max_set_diff / max_len)
    except ZeroDivisionError:
        return 0.0
