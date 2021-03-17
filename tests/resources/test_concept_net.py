import pytest

import textacy
import textacy.resources


RESOURCE = textacy.resources.ConceptNet()

pytestmark = pytest.mark.skipif(
    RESOURCE.filepath is None,
    reason="ConceptNet resource must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def spacy_doc():
    text = "The quick brown fox jumps over the lazy dog."
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.mark.skip("No need to download a new resource every time")
def test_download(tmpdir):
    resource = textacy.resources.ConceptNet(data_dir=str(tmpdir))
    resource.download()
    assert resource._filepath.is_file()


def test_oserror(tmpdir):
    resource = textacy.resources.ConceptNet(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = resource.antonyms


class TestGetRelationValues:

    def test_empty_results(self):
        invalid_senses = ["x", "y", "PROPN", "PUNCT"]
        for sense in invalid_senses:
            assert RESOURCE.get_antonyms("love", lang="en", sense=sense) == []
        missing_terms = ["foobarbatbaz", "burton dewilde", "natural language processing"]
        for term in missing_terms:
            assert RESOURCE.get_antonyms(term, lang="en", sense="n") == []

    def test_value_errors(self):
        bad_tls = [
            ("love", None, None),
            ("love", "en", None),
            ("love", None, "n"),
            ("love", "un", "n"),
        ]
        for term, lang, sense in bad_tls:
            with pytest.raises(ValueError):
                RESOURCE._get_relation_values(
                    RESOURCE.antonyms, term, lang=lang, sense=sense)

    def test_type_errors(self, spacy_doc):
        bad_tls = [
            (1, "en", "n"),
            (None, "en", "v"),
            (spacy_doc, "en", "a"),
        ]
        for term, lang, sense in bad_tls:
            with pytest.raises(TypeError):
                RESOURCE._get_relation_values(
                    RESOURCE.antonyms, term, lang=lang, sense=sense)


class TestAntonyms:

    def test_property(self):
        assert isinstance(RESOURCE.antonyms, dict)
        assert isinstance(RESOURCE.antonyms["en"], dict)
        assert isinstance(RESOURCE.antonyms["en"]["love"], dict)
        assert isinstance(RESOURCE.antonyms["en"]["love"]["v"], list)
        assert isinstance(RESOURCE.antonyms["en"]["love"]["v"][0], str)

    def test_method(self, spacy_doc):
        for tok in spacy_doc:
            assert isinstance(RESOURCE.get_antonyms(tok), list)
            assert isinstance(
                RESOURCE.get_antonyms(tok.text, lang="en", sense=tok.pos_), list)


class TestHyponyms:

    def test_property(self):
        assert isinstance(RESOURCE.hyponyms, dict)
        assert isinstance(RESOURCE.hyponyms["en"], dict)
        assert isinstance(RESOURCE.hyponyms["en"]["love"], dict)
        assert isinstance(RESOURCE.hyponyms["en"]["love"]["n"], list)
        assert isinstance(RESOURCE.hyponyms["en"]["love"]["n"][0], str)

    def test_method(self, spacy_doc):
        for tok in spacy_doc:
            assert isinstance(RESOURCE.get_hyponyms(tok), list)
            assert isinstance(
                RESOURCE.get_hyponyms(tok.text, lang="en", sense=tok.pos_), list)


class TestMeronyms:

    def test_property(self):
        assert isinstance(RESOURCE.meronyms, dict)
        assert isinstance(RESOURCE.meronyms["en"], dict)
        assert isinstance(RESOURCE.meronyms["en"]["ring"], dict)
        assert isinstance(RESOURCE.meronyms["en"]["ring"]["n"], list)
        assert isinstance(RESOURCE.meronyms["en"]["ring"]["n"][0], str)

    def test_method(self, spacy_doc):
        for tok in spacy_doc:
            assert isinstance(RESOURCE.get_meronyms(tok), list)
            assert isinstance(
                RESOURCE.get_meronyms(tok.text, lang="en", sense=tok.pos_), list)


class TestSynonyms:

    def test_property(self):
        assert isinstance(RESOURCE.synonyms, dict)
        assert isinstance(RESOURCE.synonyms["en"], dict)
        assert isinstance(RESOURCE.synonyms["en"]["love"], dict)
        assert isinstance(RESOURCE.synonyms["en"]["love"]["n"], list)
        assert isinstance(RESOURCE.synonyms["en"]["love"]["n"][0], str)

    def test_method(self, spacy_doc):
        for tok in spacy_doc:
            assert isinstance(RESOURCE.get_synonyms(tok), list)
            assert isinstance(
                RESOURCE.get_synonyms(tok.text, lang="en", sense=tok.pos_), list)
