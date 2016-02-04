"""
Set of small utility functions that do mathy stuff.
"""
from __future__ import division

import numpy as np

# TODO: make this module actually good and useful


def cosine_similarity(vec1, vec2):
    """
    Return the cosine similarity between two vectors.

    Args:
        vec1 (:class:`numpy.array`)
        vec2 (:class:`numpy.array`)

    Returns:
        float
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def levenshtein_distance(str1, str2):
    """
    Function to find the Levenshtein distance between two words/sentences;
    gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python

    Args:
        str1 (str)
        str2 (str)

    Returns:
        int
    """
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    return distances[-1]
