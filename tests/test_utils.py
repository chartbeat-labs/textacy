from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from textacy import utils


def test_to_collection():
    in_outs = [
        [(1, int, list), [1]],
        [([1, 2], int, tuple), (1, 2)],
        [((1, 1.0), (int, float), set), {1, 1.0}],
    ]
    assert utils.to_collection(None, int, list) is None
    for in_, out_ in in_outs:
        assert utils.to_collection(*in_) == out_
