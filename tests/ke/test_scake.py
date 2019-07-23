import pytest

import textacy
from textacy import datasets, ke


@pytest.fixture(scope="module")
def spacy_doc():
    ds = datasets.CapitolWords()
    text = next(ds.texts(min_len=1500, limit=1))
    return textacy.make_spacy_doc(text, lang="en")


@pytest.fixture(scope="module")
def empty_spacy_doc():
    return textacy.make_spacy_doc("", lang="en")


def test_default(spacy_doc):
    result = ke.scake(spacy_doc)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(
        isinstance(ts[0], str) and isinstance(ts[1], float)
        for ts in result
    )


def test_include_pos(spacy_doc):
    result1 = ke.scake(spacy_doc, include_pos={"NOUN", "PROPN", "ADJ"})
    result2 = ke.scake(spacy_doc, include_pos={"NOUN", "PROPN"})
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_n_topn(spacy_doc):
    for n in (5, 25):
        result = ke.scake(spacy_doc, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(spacy_doc):
    result = ke.scake(spacy_doc, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = ke.scake(spacy_doc, topn=2.0)


def test_empty_doc(empty_spacy_doc):
    result = ke.scake(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0
