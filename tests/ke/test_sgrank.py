from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

import textacy
from textacy import compat, datasets, ke


@pytest.fixture(scope="module")
def spacy_doc():
    ds = datasets.CapitolWords()
    text = next(ds.texts(min_len=1500, limit=1))
    return textacy.make_spacy_doc(text, lang="en")


@pytest.fixture(scope="module")
def empty_spacy_doc():
    return textacy.make_spacy_doc("", lang="en")


def test_default(spacy_doc):
    result = ke.sgrank(spacy_doc)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(
        isinstance(ts[0], compat.unicode_) and isinstance(ts[1], float)
        for ts in result
    )


def test_ngrams_1(spacy_doc):
    result = ke.sgrank(spacy_doc, ngrams=1)
    assert len(result) > 0
    assert all(len(term.split()) == 1 for term, _ in result)


def test_ngrams_2_3(spacy_doc):
    result = ke.sgrank(spacy_doc, ngrams=(2, 3))
    assert len(result) > 0
    assert all(2 <= len(term.split()) <= 3 for term, _ in result)


def test_topn(spacy_doc):
    for n in (5, 25):
        result = ke.sgrank(spacy_doc, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(spacy_doc):
    result = ke.sgrank(spacy_doc, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = ke.sgrank(spacy_doc, topn=2.0)


def test_normalize_lower(spacy_doc):
    result = ke.sgrank(spacy_doc, normalize="lower")
    assert len(result) > 0
    assert all(term == term.lower() for term, _ in result)


def test_normalize_none(spacy_doc):
    result = ke.sgrank(spacy_doc, normalize=None)
    assert len(result) > 0
    assert any(term != term.lower() for term, _ in result)


def test_normalize_callable(spacy_doc):
    result = ke.sgrank(spacy_doc, normalize=lambda tok: tok.text.upper())
    assert len(result) > 0
    assert all(term == term.upper() for term, _ in result)


def test_window_size(spacy_doc):
    result_10 = ke.sgrank(spacy_doc, window_size=10)
    result_100 = ke.sgrank(spacy_doc, window_size=100)
    assert len(result_10) > 0 and len(result_100) > 0
    assert result_10 != result_100
    with pytest.raises(ValueError):
        _ = ke.sgrank(spacy_doc, window_size=1)


def test_empty_doc(empty_spacy_doc):
    result = ke.sgrank(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0
