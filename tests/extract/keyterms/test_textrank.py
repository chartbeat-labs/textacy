import pytest

import textacy
from textacy.extract import keyterms as kt


@pytest.fixture(scope="module")
def empty_spacy_doc(lang_en):
    return textacy.make_spacy_doc("", lang=lang_en)


def test_default(doc_long_en):
    result = kt.textrank(doc_long_en)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(isinstance(ts[0], str) and isinstance(ts[1], float) for ts in result)


def test_n_topn(doc_long_en):
    for n in (5, 25):
        result = kt.textrank(doc_long_en, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(doc_long_en):
    result = kt.textrank(doc_long_en, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = kt.textrank(doc_long_en, topn=2.0)


def test_window_size(doc_long_en):
    result1 = kt.textrank(doc_long_en, window_size=2)
    result2 = kt.textrank(doc_long_en, window_size=4)
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_edge_weighting(doc_long_en):
    result1 = kt.textrank(doc_long_en, edge_weighting="binary")
    result2 = kt.textrank(doc_long_en, edge_weighting="count")
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_position_bias(doc_long_en):
    result1 = kt.textrank(doc_long_en, position_bias=False)
    result2 = kt.textrank(doc_long_en, position_bias=True)
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_empty_doc(empty_spacy_doc):
    result = kt.yake(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0
