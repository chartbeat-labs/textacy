"""
collection of semantic distance metrics...
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections

from cytoolz import itertoolz
from fuzzywuzzy import fuzz
import numpy as np
from pyemd import emd
from sklearn.metrics import pairwise_distances
from spacy.strings import StringStore


def word_movers_distance(doc1, doc2, metric='cosine'):
    """
    Measure the semantic distance between two documents using Word Movers Distance.

    Args:
        doc1 (`TextDoc`)
        doc2 (`TextDoc`)
        metric ({'cosine', 'euclidean', 'l1', 'l2', 'manhattan'})

    Returns:
        float: distance between `doc1` and `doc2` in [0.0, 1.0], where 0.0
            indicates that the documents are the same
    """
    stringstore = StringStore()

    n = 0
    word_vecs = []
    for word in itertoolz.concatv(doc1.words(), doc2.words()):
        if word.has_vector:
            if stringstore[word.text] - 1 == n:
                word_vecs.append(word.vector)
                n += 1
    distance_mat = pairwise_distances(np.array(word_vecs), metric=metric).astype(np.double)
    distance_mat /= distance_mat.max()

    vec1 = collections.Counter(
        stringstore[word.text] - 1
        for word in doc1.words()
        if word.has_vector)
    vec1 = np.array([vec1[word_idx] for word_idx in range(len(stringstore))]).astype(np.double)
    vec1 /= vec1.sum()  # normalize word counts

    vec2 = collections.Counter(
        stringstore[word.text] - 1
        for word in doc2.words()
        if word.has_vector)
    vec2 = np.array([vec2[word_idx] for word_idx in range(len(stringstore))]).astype(np.double)
    vec2 /= vec2.sum()  # normalize word counts

    return emd(vec1, vec2, distance_mat)


def jaccard_distance(vec1, vec2, fuzzy_match=False, match_threshold=80):
    """
    Measure the semantic distance between two vectors of strings using Jaccard
    distance, with optional fuzzy matching of not-identical pairs.

    Args:
        vec1 (sequence(str))
        vec2 (sequence(str))
        fuzzy_match (bool): if True, allow for fuzzy matching in addition to the
            usual identical matching of pairs between input vectors
        match_threshold (int): value in the interval [0, 100]; fuzzy comparisons
            with a score >= this value will be considered matches

    Returns:
        float: distance between `vec1` and `vec2` in [0.0, 1.0], where 0.0
            indicates that the documents are the same
    """
    set1 = set(vec1)
    set2 = set(vec2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if fuzzy_match is True:
        for item1 in set1.difference(set2):
            if max(fuzz.token_sort_ratio(item1, item2) for item2 in set2) >= match_threshold:
                intersection += 1
        for item2 in set2.difference(set1):
            if max(fuzz.token_sort_ratio(item2, item1) for item1 in set1) >= match_threshold:
                intersection += 1

    return 1.0 - (intersection / union)
