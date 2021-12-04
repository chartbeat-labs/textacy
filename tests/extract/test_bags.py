import pytest
import spacy

import textacy
from textacy import extract


@pytest.fixture(scope="module")
def doc():
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    meta = {"author": "Gabriel García Márquez", "title": "Cien años de soledad"}
    return textacy.make_spacy_doc((text, meta), lang="en_core_web_sm")


class TestBagOfWords:
    def test_default(self, doc):
        result = extract.to_bag_of_words(doc)
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "norm_", "norm", "orth_", "orth"]
    )
    def test_by(self, by, doc):
        result = extract.to_bag_of_words(doc, by=by)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc):
        result = extract.to_bag_of_words(doc, weighting=weighting)
        assert isinstance(result, dict)
        if weighting == "freq":
            assert all(isinstance(val, float) for val in result.values())
        else:
            assert all(isinstance(val, int) for val in result.values())

    @pytest.mark.parametrize(
        "kwargs",
        [{"filter_stops": True}, {"filter_punct": True}, {"filter_nums": True}],
    )
    def test_kwargs(self, kwargs, doc):
        result = extract.to_bag_of_words(doc, **kwargs)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc):
        with pytest.raises((AttributeError, TypeError)):
            _ = extract.to_bag_of_words(doc, by=by)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_words(doc, weighting=weighting)


class TestBagOfTerms:
    def test_all_null(self, doc):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_terms(doc)

    @pytest.mark.parametrize(
        "kwargs",
        [{"ngs": 2}, {"ngs": [2, 3], "ents": True}, {"ents": True, "ncs": True}],
    )
    def test_simple_kwargs(self, kwargs, doc):
        result = extract.to_bag_of_terms(doc, **kwargs)
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "orth_", "orth"]
    )
    def test_by(self, by, doc):
        result = extract.to_bag_of_terms(doc, by=by, ngs=2)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc):
        result = extract.to_bag_of_terms(doc, weighting=weighting, ngs=2)
        assert isinstance(result, dict)
        if weighting == "freq":
            assert all(isinstance(val, float) for val in result.values())
        else:
            assert all(isinstance(val, int) for val in result.values())

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc):
        with pytest.raises((AttributeError, TypeError)):
            _ = extract.to_bag_of_terms(doc, by=by, ngs=2)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc):
        with pytest.raises(ValueError):
            _ = extract.to_bag_of_terms(doc, weighting=weighting, ngs=2)
