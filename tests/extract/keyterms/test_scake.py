import pytest

import textacy
from textacy.extract import keyterms as kt


@pytest.fixture(scope="module")
def empty_spacy_doc(lang_en):
    return textacy.make_spacy_doc("", lang=lang_en)


def test_default(doc_long_en):
    result = kt.scake(doc_long_en)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(isinstance(ts[0], str) and isinstance(ts[1], float) for ts in result)


def test_include_pos(doc_long_en):
    result1 = kt.scake(doc_long_en, include_pos={"NOUN", "PROPN", "ADJ"})
    result2 = kt.scake(doc_long_en, include_pos={"NOUN", "PROPN"})
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_n_topn(doc_long_en):
    for n in (5, 25):
        result = kt.scake(doc_long_en, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(doc_long_en):
    result = kt.scake(doc_long_en, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = kt.scake(doc_long_en, topn=2.0)


def test_empty_doc(empty_spacy_doc):
    result = kt.scake(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0


def test_single_sentence_doc(lang_en):
    doc = textacy.make_spacy_doc(
        "This is a document with a single sentence.", lang=lang_en
    )
    result = kt.scake(doc)
    assert isinstance(result, list)
    assert len(result) > 0
