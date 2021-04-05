"""
Edit-based Similarity
---------------------

:mod:`textacy.similarity_.edits`: Normalized similarity metrics built on edit-based
algorithms that compute the number of operations (additions, subtractions, ...)
needed to transform one string into another.
"""
from __future__ import annotations

from typing import Optional

import sklearn.feature_extraction
import sklearn.metrics
from jellyfish import hamming_distance as _hamming, levenshtein_distance as _levenshtein

from .. import constants


def _shortcut(str1: str, str2: str) -> Optional[float]:
    if not str1 or not str2:
        return 0.0
    elif str1 == str2:
        return 1.0
    else:
        return None


def hamming(str1: str, str2: str) -> float:
    """
    Compute the similarity between two strings using Hamming distance,
    which gives the number of characters at corresponding string indices that differ,
    including chars in the longer string that have no correspondents in the shorter.

    Args:
        str1
        str2

    Returns:
        Similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings
    """
    distance = _hamming(str1, str2)
    max_len = max(len(str1), len(str2))
    try:
        return 1.0 - (distance / max_len)
    except ZeroDivisionError:
        return 0.0


def levenshtein(str1: str, str2: str) -> float:
    """
    Measure the similarity between two strings using Levenshtein distance,
    which gives the minimum number of character insertions, deletions, and substitutions
    needed to change one string into the other.

    Args:
        str1
        str2

    Returns:
        Similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings
    """
    distance = _levenshtein(str1, str2)
    max_len = max(len(str1), len(str2))
    try:
        return 1.0 - (distance / max_len)
    except ZeroDivisionError:
        return 0.0


def character_ngrams(str1: str, str2: str) -> float:
    """
    Measure the similarity between two strings using a character ngrams similarity
    metric, in which strings are transformed into trigrams of alnum-only characters,
    vectorized and weighted by tf-idf, then compared by cosine similarity.

    Args:
        str1
        str2

    Returns:
        Similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings

    Note:
        This method has been used in cross-lingual plagiarism detection and
        authorship attribution, and seems to work better on longer texts.
        At the very least, it is *slow* on shorter texts relative to the other
        similarity measures.
    """
    # TODO: figure out where this method falls -- not edit-based??
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        preprocessor=lambda s: " ".join(constants.RE_ALNUM.findall(s.lower())),
        analyzer="char",
        token_pattern="(?u)\\b\\w+\\b",
        ngram_range=(3, 3),
    )
    mat = vectorizer.fit_transform([str1, str2])
    return sklearn.metrics.pairwise.cosine_similarity(mat[0, :], mat[1, :]).item()
