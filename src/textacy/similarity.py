"""
Similarity
----------

Collection of semantic + lexical similarity metrics between tokens, strings,
and sequences thereof, returning values between 0.0 (totally dissimilar)
and 1.0 (totally similar).
"""
import collections
import re
from typing import Dict, Sequence, Union

import numpy as np
import sklearn.feature_extraction
from cytoolz import itertoolz
from jellyfish import levenshtein_distance as _levenshtein
from pyemd import emd
from sklearn.metrics import pairwise_distances
from spacy.tokens import Doc, Span, Token

from . import extract


RE_ALNUM = re.compile(r"[^\W_]+", flags=re.IGNORECASE | re.UNICODE)


def word_movers(doc1: Doc, doc2: Doc, metric: str = "cosine") -> float:
    """
    Measure the semantic similarity between two documents using Word Movers Distance.

    Args:
        doc1
        doc2
        metric ({"cosine", "euclidean", "l1", "l2", "manhattan"})

    Returns:
        Similarity between ``doc1`` and ``doc2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar documents.

    References:
        - Ofir Pele and Michael Werman, "A linear time histogram metric for improved
          SIFT matching," in Computer Vision - ECCV 2008, Marseille, France, 2008.
        - Ofir Pele and Michael Werman, "Fast and robust earth mover's distances,"
          in Proc. 2009 IEEE 12th Int. Conf. on Computer Vision, Kyoto, Japan, 2009.
        - Kusner, Matt J., et al. "From word embeddings to document distances."
          Proceedings of the 32nd International Conference on Machine Learning
          (ICML 2015). 2015. http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
    """
    word_idxs: Dict[str, int] = dict()

    n = 0
    word_vecs = []
    for word in itertoolz.concatv(extract.words(doc1), extract.words(doc2)):
        if word.has_vector and word_idxs.setdefault(word.orth, n) == n:
            word_vecs.append(word.vector)
            n += 1
    distance_mat = pairwise_distances(np.array(word_vecs), metric=metric).astype(
        np.double
    )
    distance_mat /= distance_mat.max()

    vec1 = collections.Counter(
        word_idxs[word.orth] for word in extract.words(doc1) if word.has_vector
    )
    vec1 = np.array([vec1[word_idx] for word_idx in range(len(word_idxs))]).astype(
        np.double
    )
    vec1 /= vec1.sum()  # normalize word counts

    vec2 = collections.Counter(
        word_idxs[word.orth] for word in extract.words(doc2) if word.has_vector
    )
    vec2 = np.array([vec2[word_idx] for word_idx in range(len(word_idxs))]).astype(
        np.double
    )
    vec2 /= vec2.sum()  # normalize word counts

    return 1.0 - emd(vec1, vec2, distance_mat)


def word2vec(obj1: Union[Doc, Span, Token], obj2: Union[Doc, Span, Token]) -> float:
    """
    Measure the semantic similarity between one spacy Doc, Span, Token, or Lexeme
    and another like object using the cosine distance between the objects' (average)
    word2vec vectors.

    Args:
        obj1
        obj2

    Returns
        Similarity between ``obj1`` and ``obj2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar objects
    """
    return obj1.similarity(obj2)


def jaccard(
    obj1: Union[str, Sequence[str]],
    obj2: Union[str, Sequence[str]],
    fuzzy_match: bool = False,
    match_threshold: float = 0.8,
) -> float:
    """
    Measure the similarity between two strings or sequences of strings using
    Jaccard distance, with optional fuzzy matching of not-identical pairs
    when ``obj1`` and ``obj2`` are sequences of strings.

    Args:
        obj1
        obj2: If str, both inputs are treated as sequences of *characters*,
            in which case fuzzy matching is not permitted
        fuzzy_match: If True, allow for fuzzy matching in addition to the
            usual identical matching of pairs between input vectors
        match_threshold: Value in the interval [0.0, 1.0]; fuzzy comparisons
            with a score >= this value will be considered matches

    Returns:
        Similarity between ``obj1`` and ``obj2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings or sequences of strings

    Raises:
        ValueError: if ``fuzzy_match`` is True but ``obj1`` and ``obj2`` are strings,
        or if ``match_threshold`` is not a valid float
    """
    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError(
            "match_threshold={} is invalid; "
            "it must be a float in the interval [0.0, 1.0]".format(match_threshold)
        )
    set1 = set(obj1)
    set2 = set(obj2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if (
        fuzzy_match is True
        and not isinstance(obj1, (str, bytes))
        and not isinstance(obj2, (str, bytes))
    ):
        for item1 in set1.difference(set2):
            if max(token_sort_ratio(item1, item2) for item2 in set2) >= match_threshold:
                intersection += 1
        for item2 in set2.difference(set1):
            if max(token_sort_ratio(item2, item1) for item1 in set1) >= match_threshold:
                intersection += 1
    elif fuzzy_match is True:
        raise ValueError("fuzzy matching not possible with str inputs")

    return intersection / union


def levenshtein(str1: str, str2: str) -> float:
    """
    Measure the similarity between two strings using Levenshtein distance, which
    gives the minimum number of character insertions, deletions, and substitutions
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
    return 1.0 - pairwise_distances(mat[0, :], mat[1, :], metric="cosine").item()
