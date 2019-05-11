"""
Similarity
----------

Collection of various semantic similarity metrics between docs, sequences of tokens,
and individual tokens, where tokens may either be spaCy objects or strings.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import re
import warnings

import numpy as np
from cytoolz import itertoolz
from Levenshtein import (
    distance as _levenshtein,
    hamming as _hamming,
    jaro_winkler as _jaro_winkler,
    ratio as _ratio,
)
from pyemd import emd
from sklearn.metrics import pairwise_distances

from . import compat
from . import extract


RE_NONWORDCHARS = re.compile(r"\W+", flags=re.IGNORECASE | re.UNICODE)


def word_movers(doc1, doc2, metric="cosine"):
    """
    Measure the semantic similarity between two documents using Word Movers
    Distance.

    Args:
        doc1 (:class:`spacy.tokens.Doc`)
        doc2 (:class:`spacy.tokens.Doc`)
        metric ({'cosine', 'euclidean', 'l1', 'l2', 'manhattan'})

    Returns:
        float: Similarity between ``doc1`` and ``doc2`` in the interval [0.0, 1.0],
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
    word_idxs = dict()

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
    vec1 = np.array(
        [vec1[word_idx] for word_idx in compat.range_(len(word_idxs))]
    ).astype(np.double)
    vec1 /= vec1.sum()  # normalize word counts

    vec2 = collections.Counter(
        word_idxs[word.orth] for word in extract.words(doc2) if word.has_vector
    )
    vec2 = np.array(
        [vec2[word_idx] for word_idx in compat.range_(len(word_idxs))]
    ).astype(np.double)
    vec2 /= vec2.sum()  # normalize word counts

    return 1.0 - emd(vec1, vec2, distance_mat)


def word2vec(obj1, obj2):
    """
    Measure the semantic similarity between one spacy Doc, Span, Token,
    or Lexeme and another like object using the cosine distance between the
    objects' (average) word2vec vectors.

    Args:
        obj1 (:class:`spacy.tokens.Doc`, :class:`spacy.tokens.Span`, :class:`spacy.tokens.Token`, or ``spacy.Lexeme``)
        obj2 (:class:`spacy.tokens.Doc`, :class:`spacy.tokens.Span`, :class:`spacy.tokens.Token`, or ``spacy.Lexeme``)

    Returns
        float: similarity between ``obj1`` and ``obj2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar objects
    """
    return obj1.similarity(obj2)


def jaccard(obj1, obj2, fuzzy_match=False, match_threshold=0.8):
    """
    Measure the semantic similarity between two strings or sequences of strings
    using Jaccard distance, with optional fuzzy matching of not-identical pairs
    when ``obj1`` and ``obj2`` are sequences of strings.

    Args:
        obj1 (str or Sequence[str])
        obj2 (str or Sequence[str]): if str, both inputs are treated as sequences
            of *characters*, in which case fuzzy matching is not permitted
        fuzzy_match (bool): if True, allow for fuzzy matching in addition to the
            usual identical matching of pairs between input vectors
        match_threshold (float): value in the interval [0.0, 1.0]; fuzzy comparisons
            with a score >= this value will be considered matches

    Returns:
        float: similarity between ``obj1`` and ``obj2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings or sequences
        of strings

    Raises:
        ValueError: if ``fuzzy_match`` is True but ``obj1`` and ``obj2`` are strings
    """
    if isinstance(match_threshold, int) and 1 <= match_threshold <= 100:
        warnings.warn(
            "`match_threshold` should be a float in [0.0, 1.0]; "
            "it was automatically converted from the provided int in [0, 100]"
        )
        match_threshold /= 100
    set1 = set(obj1)
    set2 = set(obj2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if (
        fuzzy_match is True
        and not isinstance(obj1, compat.string_types)
        and not isinstance(obj2, compat.string_types)
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


def hamming(str1, str2):
    """
    Measure the similarity between two strings using Hamming distance, which
    simply gives the number of characters in the strings that are different i.e.
    the number of substitution edits needed to change one string into the other.

    Args:
        str1 (str)
        str2 (str)

    Returns:
        float: similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings

    Note:
        This uses a *modified* Hamming distance in that it permits strings
        of different lengths to be compared.
    """
    len_str1 = len(str1)
    len_str2 = len(str2)
    if len_str1 == len_str2:
        distance = _hamming(str1, str2)
    else:
        # make sure str1 is as long as or longer than str2
        if len_str2 > len_str1:
            str1, str2 = str2, str1
            len_str1, len_str2 = len_str2, len_str1
        # distance is # of different chars + difference in str lengths
        distance = len_str1 - len_str2
        distance += _hamming(str1[:len_str2], str2)
    distance /= len_str1
    return 1.0 - distance


def levenshtein(str1, str2):
    """
    Measure the similarity between two strings using Levenshtein distance, which
    gives the minimum number of character insertions, deletions, and substitutions
    needed to change one string into the other.

    Args:
        str1 (str)
        str2 (str)
        normalize (bool): if True, divide Levenshtein distance by the total number
            of characters in the longest string; otherwise leave the distance as-is

    Returns:
        float: similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings
    """
    distance = _levenshtein(str1, str2)
    distance /= max(len(str1), len(str2))
    return 1.0 - distance


def jaro_winkler(str1, str2, prefix_weight=0.1):
    """
    Measure the similarity between two strings using Jaro-Winkler similarity
    metric, a modification of Jaro metric giving more weight to a shared prefix.

    Args:
        str1 (str)
        str2 (str)
        prefix_weight (float): the inverse value of common prefix length needed
            to consider the strings identical

    Returns:
        float: similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings
    """
    return _jaro_winkler(str1, str2, prefix_weight)


def token_sort_ratio(str1, str2):
    """
    Measure of similarity between two strings based on minimal edit distance,
    where ordering of words in each string is normalized before comparing.

    Args:
        str1 (str)
        str2 (str)

    Returns:
        float: similarity between ``str1`` and ``str2`` in the interval [0.0, 1.0],
        where larger values correspond to more similar strings.
    """
    if not str1 or not str2:
        return 0
    str1 = compat.to_unicode(str1)
    str2 = compat.to_unicode(str2)
    str1_proc = _process_and_sort(str1)
    str2_proc = _process_and_sort(str2)
    return _ratio(str1_proc, str2_proc)


def _process_and_sort(s):
    """Return a processed string with tokens sorted then re-joined."""
    return " ".join(sorted(_process(s).split()))


def _process(s):
    """
    Remove all characters but letters and numbers, strip whitespace,
    and force everything to lower-case.
    """
    if not s:
        return ""
    return RE_NONWORDCHARS.sub(" ", s).lower().strip()
