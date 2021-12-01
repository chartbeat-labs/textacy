import pytest

import textacy
import textacy.text_stats


@pytest.fixture(scope="module")
def doc():
    text = "I write code. They wrote books."
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


def test_morph(doc):
    exp = {
        "Case": {"Nom": 2},
        "Number": {"Sing": 2, "Plur": 2},
        "Person": {"1": 1, "3": 1},
        "PronType": {"Prs": 2},
        "Tense": {"Pres": 1, "Past": 1},
        "VerbForm": {"Fin": 2},
        "PunctType": {"Peri": 2},
    }
    assert textacy.text_stats.counts.morph(doc) == exp


def test_tag(doc):
    exp = {"PRP": 2, "VBP": 1, "NN": 1, ".": 2, "VBD": 1, "NNS": 1}
    assert textacy.text_stats.counts.tag(doc) == exp


def test_pos(doc):
    exp = {"PRON": 2, "VERB": 2, "NOUN": 2, "PUNCT": 2}
    assert textacy.text_stats.counts.pos(doc) == exp


def test_dep(doc):
    exp = {"nsubj": 2, "ROOT": 2, "dobj": 2, "punct": 2}
    assert textacy.text_stats.counts.dep(doc) == exp
