import os

import pytest

from textacy import load_spacy_lang
from textacy.spacier import components

TEXT = (
    "The year was 2081, and everybody was finally equal. "
    "They weren't only equal before God and the law. "
    "They were equal every which way."
)


@pytest.fixture(scope="module")
def spacy_lang():
    spacy_lang = load_spacy_lang("en")
    text_stats_component = components.TextStatsComponent()
    spacy_lang.add_pipe(text_stats_component, after="parser")

    yield spacy_lang

    # remove component after running these tests
    spacy_lang.remove_pipe("textacy_text_stats")


@pytest.fixture(scope="module")
def spacy_doc(spacy_lang):
    spacy_doc = spacy_lang(TEXT)
    return spacy_doc


def test_component_name(spacy_lang):
    assert spacy_lang.has_pipe("textacy_text_stats") is True


def test_component_attrs():
    attrs_args = [
        None,
        "flesch_kincaid_grade_level",
        ["flesch_kincaid_grade_level", "flesch_reading_ease"],
    ]
    for attrs in attrs_args:
        text_stats_component = components.TextStatsComponent(attrs=attrs)
        assert isinstance(text_stats_component.attrs, tuple) is True


def test_attrs_on_doc(spacy_lang, spacy_doc):
    tsc = spacy_lang.get_pipe("textacy_text_stats")
    for attr in tsc.attrs:
        assert spacy_doc._.has(attr) is True
        assert isinstance(spacy_doc._.get(attr), (int, float, tuple)) is True
