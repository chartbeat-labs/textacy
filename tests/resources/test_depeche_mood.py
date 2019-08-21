import pytest

import textacy
import textacy.resources


RESOURCE = textacy.resources.DepecheMood(lang="en", word_rep="lemmapos")

pytestmark = pytest.mark.skipif(
    RESOURCE.filepath is None,
    reason="DepecheMood resource must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def spacy_doc():
    text = "A friend is someone who knows all about you and still loves you."
    return textacy.make_spacy_doc(text, lang="en")


@pytest.mark.skip("No need to download a new resource every time")
def test_download(tmpdir):
    resource = textacy.resources.DepecheMood(data_dir=str(tmpdir))
    resource.download()
    assert resource._filepath.is_file()


def test_oserror(tmpdir):
    resource = textacy.resources.DepecheMood(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = resource.weights


class TestInitParams:

    def test_lang(self):
        for lang in ["en", "it"]:
            _ =  textacy.resources.DepecheMood(lang=lang, word_rep="token")
        for lang in ["es", None]:
            with pytest.raises(ValueError):
                _ =  textacy.resources.DepecheMood(lang=lang)

    def test_word_rep(self):
        for word_rep in ["token", "lemma"]:
            _ =  textacy.resources.DepecheMood(lang="it", word_rep=word_rep)
        for word_rep in ["lower", None]:
            with pytest.raises(ValueError):
                _ =  textacy.resources.DepecheMood(word_rep=word_rep)


class TestGetEmotionalVariance:

    def test_str(self):
        result = RESOURCE.get_emotional_valence("love#n")
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(val, float) for val in result.values())
        assert all(0 <= val <= 1.0 for val in result.values())

    def test_sequence_str(self):
        result = RESOURCE.get_emotional_valence(["love#n", "hate#n"])
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(val, float) for val in result.values())
        assert all(0 <= val <= 1.0 for val in result.values())

    def test_token(self, spacy_doc):
        for tok in spacy_doc:
            result = RESOURCE.get_emotional_valence(tok)
            assert isinstance(result, dict)
            assert all(isinstance(key, str) for key in result.keys())
            assert all(isinstance(val, float) for val in result.values())
            assert all(0 <= val <= 1.0 for val in result.values())

    def test_span(self, spacy_doc):
        result = RESOURCE.get_emotional_valence(spacy_doc[:8])
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(val, float) for val in result.values())
        assert all(0 <= val <= 1.0 for val in result.values())

    def test_doc(self, spacy_doc):
        result = RESOURCE.get_emotional_valence(spacy_doc)
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(val, float) for val in result.values())
        assert all(0 <= val <= 1.0 for val in result.values())
