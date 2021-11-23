import pytest

from textacy import load_spacy_lang
from textacy import text_stats


@pytest.fixture(scope="module")
def spacy_lang():
    spacy_lang = load_spacy_lang("en_core_web_sm")
    spacy_lang.add_pipe("textacy_text_stats", last=True)

    yield spacy_lang

    # remove component after running these tests
    spacy_lang.remove_pipe("textacy_text_stats")


@pytest.fixture(scope="module")
def spacy_doc(spacy_lang):
    text = (
        "The year was 2081, and everybody was finally equal. "
        "They weren't only equal before God and the law. "
        "They were equal every which way."
    )
    spacy_doc = spacy_lang(text)
    return spacy_doc


def test_component_name(spacy_lang):
    assert spacy_lang.has_pipe("textacy_text_stats") is True


def test_component_problems(spacy_lang):
    assert spacy_lang.analyze_pipes()["problems"]["textacy_text_stats"] == []


def test_doc_extension(spacy_doc):
    assert spacy_doc.has_extension("text_stats")
    assert isinstance(spacy_doc._.text_stats, text_stats.TextStats)
    assert isinstance(spacy_doc._.text_stats.n_words, int)
