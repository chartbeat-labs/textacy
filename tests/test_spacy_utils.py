from __future__ import absolute_import, unicode_literals

import pytest

from textacy import cache, spacy_utils


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = cache.load_spacy("en")
    text = """
    The unit tests aren't going well.
    I love Python, but I don't love backwards incompatibilities.
    No programmers were permanently damaged for textacy's sake.
    Thank God for Stack Overflow."""
    spacy_doc = spacy_lang(text.strip())
    return spacy_doc


def test_is_plural_noun(spacy_doc):
    plural_nouns = [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [int(spacy_utils.is_plural_noun(tok)) for tok in spacy_doc] == plural_nouns


def test_is_negated_verb(spacy_doc):
    negated_verbs = [
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [int(spacy_utils.is_negated_verb(tok)) for tok in spacy_doc] == negated_verbs


def test_preserve_case(spacy_doc):
    preserved_cases = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
    ]
    assert [int(spacy_utils.preserve_case(tok)) for tok in spacy_doc] == preserved_cases


def test_normalize_str(spacy_doc):
    normalized_strs = [
        "the",
        "unit",
        "test",
        "be",
        "not",
        "go",
        "well",
        ".",
        "-PRON-",
        "love",
        "Python",
        ",",
        "but",
        "-PRON-",
        "do",
        "not",
        "love",
        "backwards",
        "incompatibility",
        ".",
        "no",
        "programmer",
        "be",
        "permanently",
        "damage",
        "for",
        "textacy",
        "'s",
        "sake",
        ".",
        "thank",
        "God",
        "for",
        "Stack",
        "Overflow",
        ".",
    ]
    assert [
        spacy_utils.normalized_str(tok) for tok in spacy_doc if not tok.is_space
    ] == normalized_strs
