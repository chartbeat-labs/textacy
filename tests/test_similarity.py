import pytest

from textacy import similarity
from textacy import make_spacy_doc


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


class TestWordMovers:

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


class TestWord2Vec:

    def test_default(self, doc_pairs):
        for doc1, doc2 in doc_pairs:
            assert 0.0 <= similarity.word2vec(doc1, doc2) <= 1.0

    def test_identity(self, doc_pairs):
        for doc1, doc2 in doc_pairs:
            assert similarity.word2vec(doc1, doc1) == pytest.approx(1.0, rel=1e-3)
            assert similarity.word2vec(doc2, doc2) == pytest.approx(1.0, rel=1e-3)


class TestJaccard:

    def test_obj_types(self, text_pairs):
        for text1, text2 in text_pairs:
            _ = similarity.jaccard(text1, text2)
            _ = similarity.jaccard(text1.split(), text2.split())

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

    def test_fuzzy_match_error(self, text_pairs):
        for text1, text2 in text_pairs:
            with pytest.raises(ValueError):
                _ = similarity.jaccard(text1, text2, True)

    def test_match_threshold_error(self, text_pairs):
        text1, text2 = text_pairs[0]
        for mt in (-1.0, 1.01, 50):
            with pytest.raises(ValueError):
                _ = similarity.jaccard(text1, text2, match_threshold=mt)


class TestLevenshtein:

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


class TestCharacterNgrams:

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


class TestTokenSortRatio:

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
