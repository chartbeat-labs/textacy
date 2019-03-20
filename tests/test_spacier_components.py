from __future__ import absolute_import, unicode_literals

import os

import pytest

from textacy import cache
from textacy.spacier import components

TEXT = (
    "The year was 2081, and everybody was finally equal. "
    "They weren't only equal before God and the law. "
    "They were equal every which way."
)


@pytest.fixture(scope="module")
def spacy_lang():
    spacy_lang = cache.load_spacy("en")
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
        assert isinstance(spacy_doc._.get(attr), (int, float, dict)) is True


def test_merge_entities(spacy_lang):
    doc1 = spacy_lang("Matthew Honnibal and Ines Montani do great work on spaCy.")
    # (temporarily) add this other component to the pipeline
    spacy_lang.add_pipe(components.merge_entities, after="ner")
    doc2 = spacy_lang("Matthew Honnibal and Ines Montani do great work on spaCy.")
    # check the key behaviors we'd expect
    assert spacy_lang.has_pipe("merge_entities") is True
    assert len(doc1) > len(doc2)
    assert any(tok.text == "Matthew Honnibal" for tok in doc2)
    assert any(tok.text == "Ines Montani" for tok in doc2)
    # now remove this component, since we don't want it elsewhere
    spacy_lang.remove_pipe("merge_entities")
