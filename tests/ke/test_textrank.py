import pytest

import textacy
from textacy import datasets, ke


@pytest.fixture(scope="module")
def spacy_doc():
    ds = datasets.CapitolWords()
    text = next(ds.texts(min_len=1500, limit=1))
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.fixture(scope="module")
def empty_spacy_doc():
    return textacy.make_spacy_doc("", lang="en_core_web_sm")


def test_default(spacy_doc):
    result = ke.textrank(spacy_doc)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(
        isinstance(ts[0], str) and isinstance(ts[1], float)
        for ts in result
    )


def test_n_topn(spacy_doc):
    for n in (5, 25):
        result = ke.textrank(spacy_doc, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(spacy_doc):
    result = ke.textrank(spacy_doc, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = ke.textrank(spacy_doc, topn=2.0)


def test_window_size(spacy_doc):
    result1 = ke.textrank(spacy_doc, window_size=2)
    result2 = ke.textrank(spacy_doc, window_size=4)
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_edge_weighting(spacy_doc):
    result1 = ke.textrank(spacy_doc, edge_weighting="binary")
    result2 = ke.textrank(spacy_doc, edge_weighting="count")
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_position_bias(spacy_doc):
    result1 = ke.textrank(spacy_doc, position_bias=False)
    result2 = ke.textrank(spacy_doc, position_bias=True)
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_empty_doc(empty_spacy_doc):
    result = ke.yake(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0
