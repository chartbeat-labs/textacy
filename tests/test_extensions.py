import pytest

import spacy

import textacy


@pytest.fixture(scope="module")
def doc():
    lang = textacy.load_spacy_lang("en_core_web_sm")
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    meta = {"author": "Gabriel García Márquez", "title": "Cien años de soledad"}
    return textacy.make_spacy_doc((text, meta), lang=lang)


def test_extensions_exist(doc):
    for name in textacy.get_doc_extensions().keys():
        assert doc.has_extension(name)


def test_set_remove_extensions(doc):
    textacy.remove_doc_extensions()
    for name in textacy.get_doc_extensions().keys():
        assert doc.has_extension(name) is False
    textacy.set_doc_extensions()
    for name in textacy.get_doc_extensions().keys():
        assert spacy.tokens.Doc.has_extension(name) is True


def test_preview(doc):
    preview = doc._.preview
    assert isinstance(preview, str)
    assert preview.startswith("Doc")


class TestMeta:

    def test_getter(self, doc):
        meta = doc._.meta
        assert meta
        assert isinstance(meta, dict)

    def test_setter(self, doc):
        meta = {"foo": "bar"}
        doc._.meta = meta
        assert doc._.meta == meta

    def test_setter_invalid(self, doc):
        with pytest.raises(TypeError):
            doc._.meta = None


def test_tokenized_text(doc):
    result = doc._.to_tokenized_text()
    assert result
    assert (
        isinstance(result, list) and
        isinstance(result[0], list) and
        isinstance(result[0][0], str)
    )
    assert len(result) == sum(1 for _ in doc.sents)


class TestBagOfWords():

    def test_default(self, doc):
        result = doc._.to_bag_of_words()
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "norm_", "norm", "orth_", "orth"]
    )
    def test_by(self, by, doc):
        result = doc._.to_bag_of_words(by=by)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc):
        result = doc._.to_bag_of_words(weighting=weighting)
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
        result = doc._.to_bag_of_words(**kwargs)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc):
        with pytest.raises((AttributeError, TypeError)):
            _ = doc._.to_bag_of_words(by=by)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc):
        with pytest.raises(ValueError):
            _ = doc._.to_bag_of_words(weighting=weighting)


class TestBagOfTerms():

    def test_all_null(self, doc):
        with pytest.raises(ValueError):
            _ = doc._.to_bag_of_terms()

    @pytest.mark.parametrize(
        "kwargs",
        [{"ngs": 2}, {"ngs": [2, 3], "ents": True}, {"ents": True, "ncs": True}],
    )
    def test_simple_kwargs(self, kwargs, doc):
        result = doc._.to_bag_of_terms(**kwargs)
        assert isinstance(result, dict)
        assert all(
            isinstance(key, str) and isinstance(val, int) for key, val in result.items()
        )

    @pytest.mark.parametrize(
        "by", ["lemma_", "lemma", "lower_", "lower", "orth_", "orth"]
    )
    def test_by(self, by, doc):
        result = doc._.to_bag_of_terms(by=by, ngs=2)
        assert isinstance(result, dict)
        if by.endswith("_"):
            assert all(isinstance(key, str) for key in result.keys())
        else:
            assert all(isinstance(key, int) for key in result.keys())

    @pytest.mark.parametrize("weighting", ["count", "freq", "binary"])
    def test_weighting(self, weighting, doc):
        result = doc._.to_bag_of_terms(weighting=weighting, ngs=2)
        assert isinstance(result, dict)
        if weighting == "freq":
            assert all(isinstance(val, float) for val in result.values())
        else:
            assert all(isinstance(val, int) for val in result.values())

    @pytest.mark.parametrize("by", ["LEMMA", spacy.attrs.LEMMA, True])
    def test_invalid_by(self, by, doc):
        with pytest.raises((AttributeError, TypeError)):
            _ = doc._.to_bag_of_terms(by=by, ngs=2)

    @pytest.mark.parametrize("weighting", ["COUNT", "frequency", True])
    def test_invalid_weighting(self, weighting, doc):
        with pytest.raises(ValueError):
            _ = doc._.to_bag_of_terms(weighting=weighting, ngs=2)
