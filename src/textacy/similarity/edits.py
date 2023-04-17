"""
Edit-based Metrics
------------------

:mod:`textacy.similarity.edits`: Normalized similarity metrics built on edit-based
algorithms that compute the number of operations (additions, subtractions, ...)
needed to transform one string into another.
"""
from __future__ import annotations

from typing import Optional

import sklearn.feature_extraction
import sklearn.metrics
from rapidfuzz.distance import Hamming, Jaro, Levenshtein

from .. import constants


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
    if not str1 and not str2:
        return 0.0

    return Hamming.normalized_similarity(s1, s2)


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
    if not str1 and not str2:
        return 0.0

    return Levenshtein.normalized_similarity(str1, str2)


def jaro(str1: str, str2: str) -> float:
    """
    Measure the similarity between two strings using Jaro (*not* Jaro-Winkler) distance,
    which searches for common characters while taking transpositions and string lengths
    into account.

    Args:
        str1
        str2

    Returns:
        Similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings
    """
    if not str1 and not str2:
        return 0.0

    return Jaro.normalized_similarity(str1, str2)


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
