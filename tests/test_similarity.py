from __future__ import absolute_import, unicode_literals

import pytest

from textacy import Doc
from textacy import compat, similarity


@pytest.fixture(scope="module")
def text1():
    return "She spoke to the assembled journalists."


@pytest.fixture(scope="module")
def text2():
    return "He chatted with the gathered press."


@pytest.fixture(scope="module")
def doc1(text1):
    return Doc(text1, lang="en")


@pytest.fixture(scope="module")
def doc2(text2):
    return Doc(text2, lang="en")


def test_word_movers_metrics(doc1, doc2):
    metrics = ("cosine", "l1", "manhattan", "l2", "euclidean")
    for metric in metrics:
        assert 0.0 <= similarity.word_movers(doc1, doc2, metric=metric) <= 1.0


def test_word_movers_identity(doc1, doc2):
    assert similarity.word_movers(doc1, doc1) == pytest.approx(1.0, rel=1e-3)


def test_word2vec(doc1, doc2):
    pairs = ((doc1, doc2), (doc1[-2:], doc2[-2:]))
    for pair in pairs:
        assert 0.0 <= similarity.word2vec(pair[0], pair[1]) <= 1.0


def test_word2vec_identity(doc1, doc2):
    assert similarity.word2vec(doc1, doc1) == pytest.approx(1.0, rel=1e-3)


def test_jaccard(text1, text2):
    pairs = ((text1, text2), (text1.split(), text2.split()))
    expected_values = (0.4583333, 0.09091)
    for pair, expected_value in zip(pairs, expected_values):
        assert similarity.jaccard(pair[0], pair[1]) == pytest.approx(
            expected_value, rel=1e-3
        )


def test_jaccard_exception(text1, text2):
    with pytest.raises(ValueError):
        _ = similarity.jaccard(text1, text2, True)


def test_jaccard_fuzzy_match(text1, text2):
    thresholds = (0.50, 0.70, 0.90)
    expected_values = (0.454546, 0.272728, 0.09091)
    for thresh, expected_value in zip(thresholds, expected_values):
        assert similarity.jaccard(
            text1.split(), text2.split(), fuzzy_match=True, match_threshold=thresh
        ) == pytest.approx(expected_value, rel=1e-3)


def test_jaccard_fuzzy_match_warning(text1, text2):
    thresh = 50
    with pytest.warns(UserWarning):
        _ = similarity.jaccard(
            text1.split(), text2.split(), fuzzy_match=True, match_threshold=thresh
        )


def test_hamming(text1, text2):
    assert similarity.hamming(text1, text2) == 0.1282051282051282


def test_levenshtein(text1, text2):
    assert similarity.levenshtein(text1, text2) == 0.3589743589743589


def test_jaro_winkler(text1, text2):
    assert similarity.jaro_winkler(text1, text2) == 0.5718004218004219
