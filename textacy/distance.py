"""
collection of semantic distance metrics...
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections

from cytoolz import itertoolz
import numpy as np
from pyemd import emd
from sklearn.metrics import pairwise_distances
from spacy.strings import StringStore


def word_movers_distance(doc1, doc2, metric='cosine'):
    """
    Args:
        doc1 (`TextDoc`)
        doc2 (`TextDoc`)
        metric (str): valid options are {'cosine', 'euclidean', 'l1', 'l2', 'manhattan'}

    Returns:
        float: distance between `doc1` and `doc2` in [0.0, 1.0];
            returns 0.0 for identical documents
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
