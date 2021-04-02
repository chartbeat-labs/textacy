import pytest
import spacy

from textacy import load_spacy_lang, make_spacy_doc


# TODO: why are we using such a long text here??
TEXT = (
    "Since the so-called \"statistical revolution\" in the late 1980s and mid 1990s, "
    "much Natural Language Processing research has relied heavily on machine learning. "
    "Formerly, many language-processing tasks typically involved the direct hand coding "
    "of rules, which is not in general robust to natural language variation. "
    "The machine-learning paradigm calls instead for using statistical inference "
    "to automatically learn such rules through the analysis of large corpora of typical "
    "real-world examples (a corpus is a set of documents, possibly with human or "
    "computer annotations). Many different classes of machine learning algorithms "
    "have been applied to NLP tasks. These algorithms take as input a large set "
    "of \"features\" that are generated from the input data. Some of the earliest-used "
    "algorithms, such as decision trees, produced systems of hard if-then rules similar "
    "to the systems of hand-written rules that were then common. Increasingly, however, "
    "research has focused on statistical models, which make soft, probabilistic "
    "decisions based on attaching real-valued weights to each input feature. "
    "Such models have the advantage that they can express the relative certainty of "
    "many different possible answers rather than only one, producing more reliable "
    "results when such a model is included as a component of a larger system."
)


@pytest.fixture(scope="module")
def en_core_web_sm():
    return load_spacy_lang("en_core_web_sm")


@pytest.fixture(scope="module")
def doc(en_core_web_sm):
    return make_spacy_doc((TEXT, {"foo": "bar!"}), lang=en_core_web_sm)


@pytest.fixture(scope="module")
def langs():
    return (
        "en_core_web_sm",
        load_spacy_lang("en_core_web_sm"),
        lambda text: "en_core_web_sm",
    )


class TestLoadSpacyLang:

    @pytest.mark.parametrize("name", ["en_core_web_sm", "es_core_news_sm"])
    def test_load_model(self, name):
        assert isinstance(load_spacy_lang(name), spacy.language.Language)

    @pytest.mark.parametrize("kwargs", [{"exclude": ("tagger", "parser", "ner")}])
    def test_load_model_kwargs(self, kwargs):
        assert isinstance(
            load_spacy_lang("en_core_web_sm", **kwargs),
            spacy.language.Language,
        )

    @pytest.mark.parametrize("kwargs", [{"exclude": ["tagger", "parser", "ner"]}])
    def test_disable_hashability(self, kwargs):
        with pytest.raises(TypeError):
            _ = load_spacy_lang("en_core_web_sm", **kwargs)

    @pytest.mark.parametrize("name", ["en", "un"])
    def test_bad_name(self, name):
        with pytest.raises(OSError):
            _ = load_spacy_lang(name)


class TestMakeSpacyDoc:

    def test_text_data(self, langs):
        text = "This is an English sentence."
        for lang in langs:
            assert isinstance(make_spacy_doc(text, lang=lang), spacy.tokens.Doc)

    def test_record_data(self, langs):
        record = ("This is an English sentence.", {"foo": "bar"})
        for lang in langs:
            assert isinstance(make_spacy_doc(record, lang=lang), spacy.tokens.Doc)

    def test_doc_data(self, langs, en_core_web_sm):
        doc = en_core_web_sm("This is an English sentence.")
        for lang in langs:
            assert isinstance(make_spacy_doc(doc, lang=lang), spacy.tokens.Doc)

    @pytest.mark.parametrize(
        "data",
        [
            b"This is an English sentence in bytes.",
            {"content": "This is an English sentence as dict value."},
            True,
        ],
    )
    def test_invalid_data(self, data, en_core_web_sm):
        with pytest.raises(TypeError):
            _ = make_spacy_doc(data, lang=en_core_web_sm)

    @pytest.mark.parametrize(
        "lang", [b"en", ["en_core_web_sm", "es_core_news_sm"], True]
    )
    def test_invalid_lang(self, lang):
        with pytest.raises(TypeError):
            _ = make_spacy_doc("This is an English sentence.", lang=lang)

    @pytest.mark.parametrize(
        "data, lang",
        [
            ("Hello, how are you my friend?", "es_core_news_sm"),
            ("Hello, how are you my friend?", lambda x: "es_core_news_sm"),
        ]
    )
    def test_invalid_data_lang_combo(self, data, lang, en_core_web_sm):
        with pytest.raises((ValueError, TypeError)):
            _ = make_spacy_doc(en_core_web_sm(data), lang=lang)
