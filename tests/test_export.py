from __future__ import absolute_import, unicode_literals

import re

import numpy as np
import pytest

from textacy import cache, export


@pytest.fixture(scope="module")
def spacy_doc():
    text = "I would have lived in peace. But my enemies brought me war."
    spacy_lang = cache.load_spacy("en")
    spacy_doc = spacy_lang(text)
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
