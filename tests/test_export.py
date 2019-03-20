from __future__ import absolute_import, unicode_literals

import re

import numpy as np
import pytest
from spacy import attrs

from textacy import cache, export


@pytest.fixture(scope="module")
def spacy_doc():
    text = "I would have lived in peace. But my enemies brought me war."
    spacy_lang = cache.load_spacy("en")
    spacy_doc = spacy_lang(text)
    cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
    values = np.array(
        [
            [13656873538139661788, 3, 426],
            [16235386156175103506, 2, 402],
            [14200088355797579614, 1, 402],
            [3822385049556375858, 0, 8206900633647566924],
            [1292078113972184607, 18446744073709551615, 440],
            [15308085513773655218, 18446744073709551615, 436],
            [12646065887601541794, 18446744073709551613, 442],
            [17571114184892886314, 3, 404],
            [4062917326063685704, 1, 437],
            [783433942507015291, 1, 426],
            [17109001835818727656, 0, 8206900633647566924],
            [13656873538139661788, 18446744073709551615, 3965108062993911700],
            [15308085513773655218, 18446744073709551614, 413],
            [12646065887601541794, 18446744073709551613, 442],
        ],
        dtype="uint64",
    )
    spacy_doc.from_array(cols, values)
    return spacy_doc


def test_write_conll(spacy_doc):
    result = export.doc_to_conll(spacy_doc)
    assert len(re.findall(r"^# sent_id \d$", result, flags=re.MULTILINE)) == 2
    assert all(
        line.count("\t") == 9
        for line in result.split("\n")
        if re.search(r"^\d+\s", line)
    )
    assert all(
        line == re.search(r"\d+\s([\w=\.\$\-]+\s?)+", line).group()
        for line in result.split("\n")
        if re.search(r"^\d+\s", line)
    )
