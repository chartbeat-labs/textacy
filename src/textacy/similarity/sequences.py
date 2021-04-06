"""
Sequence-based Metrics
----------------------

:mod:`textacy.similarity.sequences`: Normalized similarity metrics built on
sequence-based algorithms that identify and measure the subsequences common to each.
"""
import difflib
from typing import Sequence


def matching_subsequences_ratio(
    seq1: Sequence[str], seq2: Sequence[str], **kwargs
) -> float:
    """
    Measure the similarity between two sequences of strings by finding
    contiguous matching subsequences without any "junk" elements and normalizing
    by the total number of elements.

    Args:
        seq1
        seq2
        **kwargs
            isjunk: Optional[Callable[str], bool] = None
            autojunk: bool = True

    Returns:
        Similarity between ``seq1`` and ``seq2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar sequences of strings

    Reference:
        https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio
    """
    return difflib.SequenceMatcher(a=seq1, b=seq2, **kwargs).ratio()
