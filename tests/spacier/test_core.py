import pytest
import spacy

from textacy import load_spacy_lang, make_spacy_doc


TEXT = """
Since the so-called "statistical revolution" in the late 1980s and mid 1990s, much Natural Language Processing research has relied heavily on machine learning.
Formerly, many language-processing tasks typically involved the direct hand coding of rules, which is not in general robust to natural language variation. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples (a corpus is a set of documents, possibly with human or computer annotations).
Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of hand-written rules that were then common. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
""".strip()


@pytest.fixture(scope="module")
def doc(request):
    return make_spacy_doc((TEXT, {"foo": "bar!"}), lang="en_core_web_sm")


@pytest.fixture(scope="module")
def langs():
    return (
        "en_core_web_sm",
        load_spacy_lang("en_core_web_sm"),
        lambda text: "en_core_web_sm",
    )


class TestLoadSpacyLang:

    def test_load_model(self):
        for lang in ["en_core_web_sm", "es_core_news_sm"]:
            for disable in [None, ("tagger", "parser", "ner")]:
                assert isinstance(
                    load_spacy_lang(lang, disable=disable),
                    spacy.language.Language
                )

    def test_load_blank(self):
        assert isinstance(
            load_spacy_lang("ar", allow_blank=True),
            spacy.language.Language
        )

    def test_disable_hashability(self):
        with pytest.raises(TypeError):
            _ = load_spacy_lang("en_core_web_sm", disable=["tagger", "parser", "ner"])

    def test_bad_name(self):
        for name in ("unk", "un"):
            with pytest.raises(OSError):
                _ = load_spacy_lang(name)
        with pytest.raises(ImportError):
            _ = load_spacy_lang("un", allow_blank=True)


class TestMakeSpacyDoc:

    def test_text_data(self, langs):
        text = "This is an English sentence."
        assert isinstance(make_spacy_doc(text), spacy.tokens.Doc)
        for lang in langs:
            assert isinstance(make_spacy_doc(text, lang=lang), spacy.tokens.Doc)

    def test_record_data(self, langs):
        record = ("This is an English sentence.", {"foo": "bar"})
        assert isinstance(make_spacy_doc(record), spacy.tokens.Doc)
        for lang in langs:
            assert isinstance(make_spacy_doc(record, lang=lang), spacy.tokens.Doc)

    def test_doc_data(self, langs):
        spacy_lang = load_spacy_lang("en_core_web_sm")
        doc = spacy_lang("This is an English sentence.")
        assert isinstance(make_spacy_doc(doc), spacy.tokens.Doc)
        for lang in langs:
            assert isinstance(make_spacy_doc(doc, lang=lang), spacy.tokens.Doc)

    def test_invalid_data(self):
        invalid_contents = [
            b"This is an English sentence in bytes.",
            {"content": "This is an English sentence as dict value."},
            True,
        ]
        for invalid_content in invalid_contents:
            with pytest.raises(TypeError):
                _ = make_spacy_doc(invalid_content)

    def test_invalid_lang(self):
        invalid_langs = [b"en", ["en_core_web_sm", "es_core_news_sm"], True]
        for invalid_lang in invalid_langs:
            with pytest.raises(TypeError):
                _ = make_spacy_doc("This is an English sentence.", lang=invalid_lang)

    def test_invalid_data_lang_combo(self):
        spacy_lang = load_spacy_lang("en_core_web_sm")
        combos = (
            (spacy_lang("Hello, how are you my friend?"), "es"),
            (spacy_lang("Hello, how are you my friend?"), True),
            ("This is an English sentence.", True),
            (("This is an English sentence.", {"foo": "bar"}), True),
        )
        for data, lang in combos:
            with pytest.raises((ValueError, TypeError)):
                _ = make_spacy_doc(data, lang=lang)
