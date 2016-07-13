"""
Set of small utility functions that do mathy stuff.
"""
from __future__ import division

import numpy as np

# TODO: make this module actually good and useful
# UPDATE: this module is still an orphan. Burton, get on it!


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
