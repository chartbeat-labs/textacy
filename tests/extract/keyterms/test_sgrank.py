import pytest

import textacy
from textacy.extract import keyterms as kt


@pytest.fixture(scope="module")
def empty_spacy_doc(lang_en):
    return textacy.make_spacy_doc("", lang=lang_en)


def test_default(doc_long_en):
    result = kt.sgrank(doc_long_en)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(isinstance(ts[0], str) and isinstance(ts[1], float) for ts in result)


def test_ngrams_1(doc_long_en):
    result = kt.sgrank(doc_long_en, ngrams=1)
    assert len(result) > 0
    assert all(len(term.split()) == 1 for term, _ in result)


def test_ngrams_2_3(doc_long_en):
    result = kt.sgrank(doc_long_en, ngrams=(2, 3))
    assert len(result) > 0
    assert all(2 <= len(term.split()) <= 3 for term, _ in result)


def test_topn(doc_long_en):
    for n in (5, 25):
        result = kt.sgrank(doc_long_en, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(doc_long_en):
    result = kt.sgrank(doc_long_en, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = kt.sgrank(doc_long_en, topn=2.0)


def test_normalize_lower(doc_long_en):
    result = kt.sgrank(doc_long_en, normalize="lower")
    assert len(result) > 0
    assert all(term == term.lower() for term, _ in result)


def test_normalize_none(doc_long_en):
    result = kt.sgrank(doc_long_en, normalize=None)
    assert len(result) > 0
    assert any(term != term.lower() for term, _ in result)


def test_normalize_callable(doc_long_en):
    result = kt.sgrank(doc_long_en, normalize=lambda tok: tok.text.upper())
    assert len(result) > 0
    assert all(term == term.upper() for term, _ in result)


def test_window_size(doc_long_en):
    result_10 = kt.sgrank(doc_long_en, window_size=10)
    result_100 = kt.sgrank(doc_long_en, window_size=100)
    assert len(result_10) > 0 and len(result_100) > 0
    assert result_10 != result_100
    with pytest.raises(ValueError):
        _ = kt.sgrank(doc_long_en, window_size=1)


def test_empty_doc(empty_spacy_doc):
    result = kt.sgrank(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0
