"""
Collection of semantic similarity metrics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections

from cytoolz import itertoolz
from fuzzywuzzy import fuzz
from Levenshtein import (distance as _levenshtein,
                         hamming as _hamming,
                         jaro_winkler as _jaro_winkler)
import numpy as np
from pyemd import emd
from sklearn.metrics import pairwise_distances
from spacy.strings import StringStore

import textacy
from textacy import extract
from textacy.compat import string_types


def word_movers(doc1, doc2, metric='cosine'):
    """
    Measure the semantic similarity between two documents using Word Movers
    Distance.

    Args:
        doc1 (``textacy.Doc`` or ``spacy.Doc``)
        doc2 (``textacy.Doc`` or ``spacy.Doc``)
        metric ({'cosine', 'euclidean', 'l1', 'l2', 'manhattan'})

    Returns:
        float: similarity between `doc1` and `doc2` in the interval [0.0, 1.0],
            where larger values correspond to more similar documents

    References:
        Ofir Pele and Michael Werman, "A linear time histogram metric for improved
            SIFT matching," in Computer Vision - ECCV 2008, Marseille, France, 2008.
        Ofir Pele and Michael Werman, "Fast and robust earth mover's distances,"
            in Proc. 2009 IEEE 12th Int. Conf. on Computer Vision, Kyoto, Japan, 2009.
        Kusner, Matt J., et al. "From word embeddings to document distances."
            Proceedings of the 32nd International Conference on Machine Learning
            (ICML 2015). 2015. http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf
    """
    stringstore = StringStore()

    n = 0
    word_vecs = []
    for word in itertoolz.concatv(extract.words(doc1), extract.words(doc2)):
        if word.has_vector:
            if stringstore[word.text] - 1 == n:  # stringstore[0] always empty space
                word_vecs.append(word.vector)
                n += 1
    distance_mat = pairwise_distances(np.array(word_vecs), metric=metric).astype(np.double)
    distance_mat /= distance_mat.max()

    vec1 = collections.Counter(
        stringstore[word.text] - 1
        for word in extract.words(doc1)
        if word.has_vector)
    vec1 = np.array([vec1[word_idx] for word_idx in range(len(stringstore))]).astype(np.double)
    vec1 /= vec1.sum()  # normalize word counts

    vec2 = collections.Counter(
        stringstore[word.text] - 1
        for word in extract.words(doc2)
        if word.has_vector)
    vec2 = np.array([vec2[word_idx] for word_idx in range(len(stringstore))]).astype(np.double)
    vec2 /= vec2.sum()  # normalize word counts

    return 1.0 - emd(vec1, vec2, distance_mat)


def word2vec(obj1, obj2):
    """
    Measure the semantic similarity between one Doc or spacy Doc, Span, Token,
    or Lexeme and another like object using the cosine distance between the
    objects' (average) word2vec vectors.

    Args:
        obj1 (``textacy.Doc``, ``spacy.Doc``, ``spacy.Span``, ``spacy.Token``, or ``spacy.Lexeme``)
        obj2 (``textacy.Doc``, ``spacy.Doc``, ``spacy.Span``, ``spacy.Token``, or ``spacy.Lexeme``)

    Returns
        float: similarity between `obj1` and `obj2` in the interval [0.0, 1.0],
            where larger values correspond to more similar objects
    """
    if isinstance(obj1, textacy.Doc) and isinstance(obj2, textacy.Doc):
        obj1 = obj1.spacy_doc
        obj2 = obj2.spacy_doc
    return obj1.similarity(obj2)


def jaccard(obj1, obj2, fuzzy_match=False, match_threshold=80):
    """
    Measure the semantic similarity between two strings or sequences of strings
    using Jaccard distance, with optional fuzzy matching of not-identical pairs
    when `obj1` and `obj2` are sequences of strings.

    Args:
        obj1 (str or Sequence[str])
        obj2 (str or Sequence[str]): if str, both inputs are treated as sequences
            of *characters*, in which case fuzzy matching is not permitted
        fuzzy_match (bool): if True, allow for fuzzy matching in addition to the
            usual identical matching of pairs between input vectors
        match_threshold (int): value in the interval [0, 100]; fuzzy comparisons
            with a score >= this value will be considered matches

    Returns:
        float: similarity between `obj1` and `obj2` in the interval [0.0, 1.0],
            where larger values correspond to more similar strings or sequences
            of strings

    Raises:
        ValueError: if `fuzzy_match` is True but `obj1` and `obj2` are strings
    """
    set1 = set(obj1)
    set2 = set(obj2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if (fuzzy_match is True and
            not isinstance(obj1, string_types) and
            not isinstance(obj2, string_types)):
        for item1 in set1.difference(set2):
            if max(fuzz.token_sort_ratio(item1, item2) for item2 in set2) >= match_threshold:
                intersection += 1
        for item2 in set2.difference(set1):
            if max(fuzz.token_sort_ratio(item2, item1) for item1 in set1) >= match_threshold:
                intersection += 1
    elif fuzzy_match is True:
        raise ValueError('fuzzy matching not possible with str inputs')

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
        float: similarity between `str1` and `str2` in the interval [0.0, 1.0],
            where larger values correspond to more similar strings

    .. note:: This uses a *modified* Hamming distance in that it permits strings
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
        float: similarity between `str1` and `str2` in the interval [0.0, 1.0],
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
        float: similarity between `str1` and `str2` in the interval [0.0, 1.0],
            where larger values correspond to more similar strings
    """
    return _jaro_winkler(str1, str2, prefix_weight)
