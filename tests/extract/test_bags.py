import pytest
import spacy

from textacy import extract


class TestBagOfWords:
    def test_default(self, doc_en):
        result = extract.to_bag_of_words(doc_en)
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "norm_", "norm", "orth_", "orth"]
    )
    def test_by(self, by, doc_en):
        result = extract.to_bag_of_words(doc_en, by=by)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc_en):
        result = extract.to_bag_of_words(doc_en, weighting=weighting)
        assert isinstance(result, dict)
        if weighting == "freq":
            assert all(isinstance(val, float) for val in result.values())
        else:
            assert all(isinstance(val, int) for val in result.values())

    @pytest.mark.parametrize(
        "kwargs",
        [{"filter_stops": True}, {"filter_punct": True}, {"filter_nums": True}],
    )
    def test_kwargs(self, kwargs, doc_en):
        result = extract.to_bag_of_words(doc_en, **kwargs)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc_en):
        with pytest.raises((AttributeError, TypeError)):
            _ = extract.to_bag_of_words(doc_en, by=by)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc_en):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_words(doc_en, weighting=weighting)


class TestBagOfTerms:
    def test_all_null(self, doc_en):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_terms(doc_en)

    @pytest.mark.parametrize(
        "kwargs",
        [{"ngs": 2}, {"ngs": [2, 3], "ents": True}, {"ents": True, "ncs": True}],
    )
    def test_simple_kwargs(self, kwargs, doc_en):
        result = extract.to_bag_of_terms(doc_en, **kwargs)
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "orth_", "orth"]
    )
    def test_by(self, by, doc_en):
        result = extract.to_bag_of_terms(doc_en, by=by, ngs=2)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc_en):
        result = extract.to_bag_of_terms(doc_en, weighting=weighting, ngs=2)
        assert isinstance(result, dict)
        if weighting == "freq":
            assert all(isinstance(val, float) for val in result.values())
        else:
            assert all(isinstance(val, int) for val in result.values())

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc_en):
        with pytest.raises((AttributeError, TypeError)):
            _ = extract.to_bag_of_terms(doc_en, by=by, ngs=2)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc_en):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_terms(doc_en, weighting=weighting, ngs=2)
