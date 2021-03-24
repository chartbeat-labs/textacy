import pytest

import textacy
from textacy import datasets
from textacy.extract import keyterms as kt


DATASET = datasets.CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def spacy_doc():
    text = next(DATASET.texts(min_len=1500, limit=1))
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.fixture(scope="module")
def empty_spacy_doc():
    return textacy.make_spacy_doc("", lang="en_core_web_sm")


def test_default(spacy_doc):
    result = kt.scake(spacy_doc)
    assert isinstance(result, list) and len(result) > 0
    assert all(isinstance(ts, tuple) and len(ts) == 2 for ts in result)
    assert all(
        isinstance(ts[0], str) and isinstance(ts[1], float)
        for ts in result
    )


def test_include_pos(spacy_doc):
    result1 = kt.scake(spacy_doc, include_pos={"NOUN", "PROPN", "ADJ"})
    result2 = kt.scake(spacy_doc, include_pos={"NOUN", "PROPN"})
    assert len(result1) > 0 and len(result2) > 0
    assert result1 != result2


def test_n_topn(spacy_doc):
    for n in (5, 25):
        result = kt.scake(spacy_doc, topn=n)
        assert 0 < len(result) <= n


def test_topn_float(spacy_doc):
    result = kt.scake(spacy_doc, topn=0.2)
    assert len(result) > 0
    with pytest.raises(ValueError):
        _ = kt.scake(spacy_doc, topn=2.0)


def test_empty_doc(empty_spacy_doc):
    result = kt.scake(empty_spacy_doc)
    assert isinstance(result, list)
    assert len(result) == 0


def test_single_sentence_doc():
    doc = textacy.make_spacy_doc(
        "This is a document with a single sentence.",
        lang="en_core_web_sm",
    )
    result = kt.scake(doc)
    assert isinstance(result, list)
    assert len(result) > 0
