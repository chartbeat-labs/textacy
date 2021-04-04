from __future__ import annotations

import re

import sklearn.feature_extraction
import sklearn.metrics
from jellyfish import hamming_distance as _hamming, levenshtein_distance as _levenshtein


RE_ALNUM = re.compile(r"[^\W_]+")


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


def token_sort_ratio(str1: str, str2: str) -> float:
    """
    Measure the similarity between two strings based on :func:`levenshtein()`,
    only with non-alphanumeric characters removed and the ordering of words
    in each string sorted before comparison.

    Args:
        str1
        str2

    Returns:
        Similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings.
    """
    if not str1 or not str2:
        return 0.0
    str1_proc = _process_and_sort(str1)
    str2_proc = _process_and_sort(str2)
    return levenshtein(str1_proc, str2_proc)


def _process_and_sort(s: str) -> str:
    """
    Remove all characters from ``s`` except letters and numbers, strip whitespace,
    and force everything to lower-case; then sort tokens before re-joining into
    a single string.
    """
    if not s:
        return ""
    return " ".join(sorted(RE_ALNUM.findall(s.lower())))


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
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        preprocessor=lambda s: " ".join(RE_ALNUM.findall(s.lower())),
        analyzer="char",
        token_pattern="(?u)\\b\\w+\\b",
        ngram_range=(3, 3),
    )
    mat = vectorizer.fit_transform([str1, str2])
    return sklearn.metrics.pairwise.cosine_similarity(mat[0, :], mat[1, :]).item()
