# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import collections
import pytest

from spacy.tokens import Span as SpacySpan
from spacy.tokens import Token as SpacyToken

from textacy import cache, constants, extract


@pytest.fixture(scope='module')
def spacy_doc():
    spacy_lang = cache.load_spacy('en')
    text = """
    Two weeks ago, I was in Kuwait participating in an I.M.F. seminar for Arab educators. For 30 minutes, we discussed the impact of technology trends on education in the Middle East. And then an Egyptian education official raised his hand and asked if he could ask me a personal question: "I heard Donald Trump say we need to close mosques in the United States," he said with great sorrow. "Is that what we want our kids to learn?"
    """
    spacy_doc = spacy_lang(text.strip())
    return spacy_doc


def test_words(spacy_doc):
    expected = [
        'Two', 'weeks', 'ago', ',', 'I', 'was', 'in', 'Kuwait', 'participating',
        'in', 'an', 'I.M.F.', 'seminar', 'for', 'Arab', 'educators', '.', 'For',
        '30', 'minutes', ',', 'we', 'discussed', 'the', 'impact']
    observed = [
        tok.text for tok in extract.words(spacy_doc,
                                          filter_stops=False, filter_punct=False, filter_nums=False)][:25]
    assert observed == expected


def test_words_filter(spacy_doc):
    result = [
        tok for tok in extract.words(spacy_doc,
                                     filter_stops=True, filter_punct=True, filter_nums=True)]
    assert not any(tok.is_stop for tok in result)
    assert not any(tok.is_punct for tok in result)
    assert not any(tok.like_num for tok in result)


def test_words_good_tags(spacy_doc):
    result = [
        tok for tok in extract.words(spacy_doc,
                                     filter_stops=False, filter_punct=False, filter_nums=False,
                                     include_pos={'NOUN'})]
    assert all(tok.pos_ == 'NOUN' for tok in result)


def test_words_min_freq(spacy_doc):
    counts = collections.Counter()
    counts.update(tok.lower_ for tok in spacy_doc)
    result = [
        tok for tok in extract.words(spacy_doc,
                                     filter_stops=False, filter_punct=False, filter_nums=False,
                                     min_freq=2)]
    assert all(counts[tok.lower_] >= 2 for tok in result)


def test_ngrams_less_than_1(spacy_doc):
    with pytest.raises(ValueError):
        _ = list(extract.ngrams(spacy_doc, 0))


def test_ngrams_n(spacy_doc):
    for n in (1, 2):
        result = [
            span for span in extract.ngrams(spacy_doc, n,
                                            filter_stops=False, filter_punct=False, filter_nums=False)]
        assert all(len(span) == n for span in result)
        assert all(isinstance(span, SpacySpan) for span in result)


def test_ngrams_filter(spacy_doc):
    result = [
        span for span in extract.ngrams(spacy_doc, 2,
                                        filter_stops=True, filter_punct=True, filter_nums=True)]
    assert not any(span[0].is_stop or span[-1].is_stop for span in result)
    assert not any(tok.is_punct for span in result for tok in span)
    assert not any(tok.like_num for span in result for tok in span)


def test_ngrams_min_freq(spacy_doc):
    n = 2
    counts = collections.Counter()
    counts.update(spacy_doc[i: i + n].lower_
                  for i in range(len(spacy_doc) - n + 1))
    result = [span for span in extract.ngrams(spacy_doc, n,
                                              filter_stops=False, filter_punct=False, filter_nums=False,
                                              min_freq=2)]
    assert all(counts[span.lower_] >= 2 for span in result)


def test_ngrams_good_tag(spacy_doc):
    result = [
        span for span in extract.ngrams(spacy_doc, 2,
                                        filter_stops=False, filter_punct=False, filter_nums=False,
                                        include_pos={'NOUN'})]
    assert all(tok.pos_ == 'NOUN' for span in result for tok in span)


def test_named_entities(spacy_doc):
    result = [
        ent for ent in extract.named_entities(spacy_doc, drop_determiners=False)]
    assert all(ent.label_ for ent in result)
    assert all(ent[0].ent_type for ent in result)


def test_named_entities_good(spacy_doc):
    include_types = {'PERSON', 'GPE'}
    result = [
        ent for ent in extract.named_entities(spacy_doc,
                                              include_types=include_types,
                                              drop_determiners=False)]
    assert all(ent.label_ in include_types for ent in result)


def test_named_entities_min_freq(spacy_doc):
    expected = []
    observed = [
        ent.text for ent in extract.named_entities(spacy_doc,
                                                   drop_determiners=True,
                                                   min_freq=2)]
    assert observed == expected


def test_named_entities_determiner(spacy_doc):
    expected = ['the Middle East', 'the United States']
    observed = [
        ent.text for ent in extract.named_entities(spacy_doc, drop_determiners=False)
        if ent[0].pos_ == 'DET']
    assert observed == expected


def test_named_entities_drop_determiners(spacy_doc):
    ents = list(extract.named_entities(spacy_doc, drop_determiners=True))
    assert not any(ent[0].tag_ == 'DET' for ent in ents)
    assert all(ent.label_ for ent in ents)


def test_noun_chunks(spacy_doc):
    expected = [
        'I', 'Kuwait', 'I.M.F. seminar', 'Arab educators', '30 minutes', 'we',
        'impact', 'technology trends', 'education', 'Middle East']
    observed = [
        nc.text for nc in extract.noun_chunks(spacy_doc, drop_determiners=True)][:10]
    assert observed == expected


def test_noun_chunks_determiner(spacy_doc):
    expected = [
        'I', 'Kuwait', 'an I.M.F. seminar', 'Arab educators', '30 minutes', 'we',
        'the impact', 'technology trends', 'education', 'the Middle East']
    observed = [
        nc.text for nc in extract.noun_chunks(spacy_doc, drop_determiners=False)][:10]
    assert observed == expected


def test_noun_chunks_min_freq(spacy_doc):
    expected = ['I', 'we', 'he', 'I', 'we', 'he', 'we']
    observed = [
        nc.text for nc in extract.noun_chunks(spacy_doc, drop_determiners=True, min_freq=2)]
    assert observed == expected


def test_pos_regex_matches(spacy_doc):
    expected = [
        'Two weeks', 'Kuwait', 'an I.M.F. seminar', 'Arab educators',
        '30 minutes', 'the impact', 'technology trends', 'education',
        'the Middle East', 'an Egyptian education official', 'his hand',
        'a personal question', 'Donald Trump', 'mosques',
        'the United States', 'great sorrow', 'that what', 'our kids']
    observed = [
        span.text for span in extract.pos_regex_matches(spacy_doc,
                                                        constants.POS_REGEX_PATTERNS['en']['NP'])]
    assert observed == expected


def test_subject_verb_object_triples(spacy_doc):
    expected = [
        'we, discussed, impact', 'education official, raised, hand', 'he, could ask, me',
        'he, could ask, question', 'we, need, to close']
    observed = [
        ', '.join(item.text for item in triple)
        for triple in extract.subject_verb_object_triples(spacy_doc)]
    assert observed == expected


def test_acronyms_and_definitions(spacy_doc):
    expected = {'I.M.F.': ''}
    observed = extract.acronyms_and_definitions(spacy_doc)
    assert observed == expected


def test_acronyms_and_definitions_known(spacy_doc):
    expected = {'I.M.F.': 'International Monetary Fund'}
    observed = extract.acronyms_and_definitions(
        spacy_doc, known_acro_defs={'I.M.F.': 'International Monetary Fund'})
    assert observed == expected


@pytest.mark.skip(
    reason='Direct quotation extraction needs to be improved; it fails here')
def test_direct_quotations(spacy_doc):
    expected = [
        'he, said, "I heard Donald Trump say we need to close mosques in the United States,"',
        'he, said, "Is that what we want our kids to learn?"']
    observed = [
        ', '.join(item.text for item in triple)
        for triple in extract.direct_quotations(spacy_doc)]
    assert observed == expected
