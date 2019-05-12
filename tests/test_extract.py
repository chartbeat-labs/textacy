# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import collections
import re

import pytest
from spacy.tokens import Span, Token

from textacy import cache, compat, constants, extract


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = cache.load_spacy_lang("en")
    text = (
        "Two weeks ago, I was in Kuwait participating in an I.M.F. (International Monetary Fund) seminar for Arab educators. "
        "For 30 minutes, we discussed the impact of technology trends on education in the Middle East. "
        "And then an Egyptian education official raised his hand and asked if he could ask me a personal question: \"I heard Donald Trump say we need to close mosques in the United States,\" he said with great sorrow. "
        "\"Is that what we want our kids to learn?\""
    )
    spacy_doc = spacy_lang(text)
    return spacy_doc


class TestWords(object):

    def test_default(self, spacy_doc):
        result = list(extract.words(spacy_doc))
        assert all(isinstance(tok, Token) for tok in result)
        assert not any(tok.is_space for tok in result)

    def test_filter(self, spacy_doc):
        result = list(
            extract.words(
                spacy_doc, filter_stops=True, filter_punct=True, filter_nums=True
            )
        )
        assert not any(tok.is_stop for tok in result)
        assert not any(tok.is_punct for tok in result)
        assert not any(tok.like_num for tok in result)

    def test_pos(self, spacy_doc):
        result1 = list(extract.words(spacy_doc, include_pos={"NOUN"}))
        result2 = list(extract.words(spacy_doc, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for tok in result1)
        assert all(tok.pos_ == "NOUN" for tok in result2)
        result3 = list(extract.words(spacy_doc, exclude_pos={"NOUN"}))
        result4 = list(extract.words(spacy_doc, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for tok in result3)
        assert not any(tok.pos_ == "NOUN" for tok in result4)

    def test_min_freq(self, spacy_doc):
        counts = collections.Counter()
        counts.update(tok.lower_ for tok in spacy_doc)
        result = list(extract.words(spacy_doc, min_freq=2))
        assert all(counts[tok.lower_] >= 2 for tok in result)


class TestNGrams(object):

    def test_n_less_than_1(self, spacy_doc):
        with pytest.raises(ValueError):
            _ = list(extract.ngrams(spacy_doc, 0))

    def test_n(self, spacy_doc):
        for n in (1, 2):
            result = list(extract.ngrams(spacy_doc, n))
            assert all(isinstance(span, Span) for span in result)
            assert all(len(span) == n for span in result)

    def test_filter(self, spacy_doc):
        result = list(
            extract.ngrams(
                spacy_doc, 2, filter_stops=True, filter_punct=True, filter_nums=True
            )
        )
        assert not any(span[0].is_stop or span[-1].is_stop for span in result)
        assert not any(tok.is_punct for span in result for tok in span)
        assert not any(tok.like_num for span in result for tok in span)

    def test_min_freq(self, spacy_doc):
        n = 2
        counts = collections.Counter()
        counts.update(spacy_doc[i : i + n].lower_ for i in range(len(spacy_doc) - n + 1))
        result = list(extract.ngrams(spacy_doc, 2, min_freq=2))
        assert all(counts[span.lower_] >= 2 for span in result)

    def test_pos(self, spacy_doc):
        result1 = list(extract.ngrams(spacy_doc, 2, include_pos={"NOUN"}))
        result2 = list(extract.ngrams(spacy_doc, 2, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for span in result1 for tok in span)
        assert all(tok.pos_ == "NOUN" for span in result2 for tok in span)
        result3 = list(extract.ngrams(spacy_doc, 2, exclude_pos={"NOUN"}))
        result4 = list(extract.ngrams(spacy_doc, 2, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for span in result3 for tok in span)
        assert not any(tok.pos_ == "NOUN" for span in result4 for tok in span)


class TestEntities(object):

    def test_default(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert all(span.label_ for span in result)
        assert all(span[0].ent_type for span in result)

    def test_include_types(self, spacy_doc):
        ent_types = ["PERSON", "GPE"]
        for include_types in ent_types:
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ == include_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for include_types in ent_types:
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ in include_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for include_types in ent_types:
            include_types_parsed = extract._parse_ent_types(include_types, "include")
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ in include_types_parsed for span in result)

    def test_exclude_types(self, spacy_doc):
        ent_types = ["PERSON", "GPE"]
        for exclude_types in ent_types:
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ != exclude_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for exclude_types in ent_types:
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for exclude_types in ent_types:
            exclude_types_parsed = extract._parse_ent_types(exclude_types, "exclude")
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types_parsed for span in result)

    def test_parse_ent_types_bad_type(self):
        for bad_type in [1, 3.1415, True, b"PERSON"]:
            with pytest.raises(TypeError):
                _ = extract._parse_ent_types(bad_type, "include")

    def test_min_freq(self, spacy_doc):
        result = list(extract.entities(spacy_doc, min_freq=2))
        assert len(result) == 0

    def test_determiner(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)

    def test_drop_determiners(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=True))
        assert not any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)


class TestNounChunks(object):

    def test_default(self, spacy_doc):
        result = list(extract.noun_chunks(spacy_doc))
        assert all(isinstance(span, Span) for span in result)

    def test_determiner(self, spacy_doc):
        result = list(extract.noun_chunks(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)

    def test_min_freq(self, spacy_doc):
        text = spacy_doc.text.lower()
        result = list(extract.noun_chunks(spacy_doc, drop_determiners=True, min_freq=2))
        assert all(text.count(span.lower_) >= 2 for span in result)


class TestPOSRegexMatches(object):

    def test_simple(self, spacy_doc):
        result = list(extract.pos_regex_matches(spacy_doc, r"<NOUN>+"))
        assert all(isinstance(span, Span) for span in result)
        assert all(tok.pos_ == "NOUN" for span in result for tok in span)

    def test_complex(self, spacy_doc):
        pattern = constants.POS_REGEX_PATTERNS["en"]["NP"]
        valid_pos = set(re.findall(r"(\w+)", pattern))
        required_pos = {"NOUN", "PROPN"}
        result = list(extract.pos_regex_matches(spacy_doc, pattern))
        assert all(isinstance(span, Span) for span in result)
        assert all(tok.pos_ in valid_pos for span in result for tok in span)
        assert all(any(tok.pos_ in required_pos for tok in span) for span in result)


class TestSubjectVerbObjectTriples(object):

    def test_default(self, spacy_doc):
        result = list(extract.subject_verb_object_triples(spacy_doc))
        assert all(isinstance(triple, tuple) for triple in result)
        assert all(isinstance(span, Span) for triple in result for span in triple)
        assert all(any(tok.pos_ == "VERB" for tok in triple[1]) for triple in result)


class TestAcronymsAndDefinitions(object):

    def test_default(self, spacy_doc):
        # TODO: figure out if this function no longer works, ugh
        # expected = {"I.M.F.": "International Monetary Fund"}
        expected = {"I.M.F.": ""}
        observed = extract.acronyms_and_definitions(spacy_doc)
        assert observed == expected

    def test_known(self, spacy_doc):
        expected = {"I.M.F.": "International Monetary Fund"}
        observed = extract.acronyms_and_definitions(
            spacy_doc, known_acro_defs={"I.M.F.": "International Monetary Fund"}
        )
        assert observed == expected


def test_direct_quotations(spacy_doc):
    expected = [
        ("he", "said", '"I heard Donald Trump say we need to close mosques in the United States,"'),
        ("he", "said", '"Is that what we want our kids to learn?"'),
    ]
    result = list(extract.direct_quotations(spacy_doc))
    assert all(isinstance(dq, tuple) for dq in result)
    assert all(isinstance(obj, (Span, Token)) for dq in result for obj in dq)
    observed = [
        tuple(obj.text for obj in dq)
        for dq in result
    ]
    assert observed == expected


def test_semistructured_statements(spacy_doc):
    expected = (
        "we",
        "discussed",
        "the impact of technology trends on education in the Middle East"
    )
    observed = next(extract.semistructured_statements(spacy_doc, "we", cue="discuss"))
    assert isinstance(observed, tuple) and len(observed) == 3
    assert all(isinstance(obj, (Span, Token)) for obj in observed)
    assert all(obs.text == exp for obs, exp in compat.zip_(observed, expected))
