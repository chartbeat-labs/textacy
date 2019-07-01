from __future__ import absolute_import, unicode_literals

import pytest

from textacy import compat, similarity
from textacy.doc import make_spacy_doc


@pytest.fixture(scope="module")
def text_pairs():
    return [
        ("She spoke to the assembled journalists.", "He chatted with the gathered press."),
        ("The cats and dogs are playing.", "It's raining cats and dogs."),
        ("Saturday", "Sunday"),
    ]


@pytest.fixture(scope="module")
def doc_pairs(text_pairs):
    return [
        (make_spacy_doc(text1, lang="en"), make_spacy_doc(text2, lang="en"))
        for text1, text2 in text_pairs
    ]


class TestWordMovers(object):

    def test_metrics(self, doc_pairs):
        metrics = ("cosine", "l1", "manhattan", "l2", "euclidean")
        for metric in metrics:
            for doc1, doc2 in doc_pairs:
                assert 0.0 <= similarity.word_movers(doc1, doc2, metric=metric) <= 1.0

    def test_identity(self, doc_pairs):
        for doc1, doc2 in doc_pairs[:2]:  # HACK
            print(doc1, doc2)
            assert similarity.word_movers(doc1, doc1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.word_movers(doc2, doc2) == pytest.approx(1.0, rel=1e-3)


class TestWord2Vec(object):

    def test_default(self, doc_pairs):
        for doc1, doc2 in doc_pairs:
            assert 0.0 <= similarity.word2vec(doc1, doc2) <= 1.0

    def test_identity(self, doc_pairs):
        for doc1, doc2 in doc_pairs:
            assert similarity.word2vec(doc1, doc1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.word2vec(doc2, doc2) == pytest.approx(1.0, rel=1e-3)


class TestJaccard(object):

    def test_obj_types(self, text_pairs):
        for text1, text2 in text_pairs:
            _ = similarity.jaccard(text1, text2)
            _ = similarity.jaccard(text1.split(), text2.split())

    def test_exception(self, text_pairs):
        for text1, text2 in text_pairs:
            with pytest.raises(ValueError):
                _ = similarity.jaccard(text1, text2, True)

    def test_fuzzy_match(self, text_pairs):
        thresholds = (0.5, 0.7, 0.9)
        for text1, text2 in text_pairs:
            sims = [
                similarity.jaccard(
                    text1.split(), text2.split(),
                    fuzzy_match=True, match_threshold=threshold
                )
                for threshold in thresholds
            ]
            assert sims[0] >= sims[1] >= sims[2]

    def test_fuzzy_match_warning(self, text_pairs):
        threshold = 50
        for text1, text2 in text_pairs:
            with pytest.warns(UserWarning):
                _ = similarity.jaccard(
                    text1.split(), text2.split(),
                    fuzzy_match=True, match_threshold=threshold
                )


class TestHamming(object):

    def test_default(self, text_pairs):
        for text1, text2 in text_pairs:
            assert 0.0 <= similarity.hamming(text1, text2) <= 1.0

    def test_identity(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.hamming(text1, text1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.hamming(text2, text2) == pytest.approx(1.0, rel=1e-3)

    def test_empty(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.hamming(text1, "") == 0.0


class TestLevenshtein(object):

    def test_default(self, text_pairs):
        for text1, text2 in text_pairs:
            assert 0.0 <= similarity.levenshtein(text1, text2) <= 1.0

    def test_identity(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.levenshtein(text1, text1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.levenshtein(text2, text2) == pytest.approx(1.0, rel=1e-3)

    def test_empty(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.levenshtein(text1, "") == 0.0


class TestCharacterNgrams(object):

    def test_default(self, text_pairs):
        for text1, text2 in text_pairs:
            assert 0.0 <= similarity.character_ngrams(text1, text2) <= 1.0

    def test_identity(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.character_ngrams(text1, text1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.character_ngrams(text2, text2) == pytest.approx(1.0, rel=1e-3)

    def test_empty(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.character_ngrams(text1, "") == 0.0


class TestTokenSortRatio(object):

    def test_default(self, text_pairs):
        for text1, text2 in text_pairs:
            assert 0.0 <= similarity.token_sort_ratio(text1, text2) <= 1.0

    def test_identity(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.token_sort_ratio(text1, text1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.token_sort_ratio(text2, text2) == pytest.approx(1.0, rel=1e-3)

    def test_empty(self, text_pairs):
        for text1, text2 in text_pairs:
            assert similarity.token_sort_ratio(text1, "") == 0.0


def test_jaro_winkler(text_pairs):
    for text1, text2 in text_pairs:
        assert 0.0 <= similarity.jaro_winkler(text1, text2) <= 1.0
