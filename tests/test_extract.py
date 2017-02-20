# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

import numpy as np
from spacy import attrs

from textacy import constants, data, extract


class ExtractTestCase(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        spacy_pipeline = data.load_spacy('en')
        text = """
            Two weeks ago, I was in Kuwait participating in an I.M.F. seminar for Arab educators. For 30 minutes, we discussed the impact of technology trends on education in the Middle East. And then an Egyptian education official raised his hand and asked if he could ask me a personal question: "I heard Donald Trump say we need to close mosques in the United States," he said with great sorrow. "Is that what we want our kids to learn?"
            """
        self.spacy_doc = spacy_pipeline(text.strip())
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[425, 1, 1500074], [443, 1, 392], [447, 3, 365], [416, 2, 407], [445, 1, 393],
             [455, 0, 53503], [432, -1, 405], [441, -1, 401], [456, -3, 364],
             [432, -1, 405], [426, 2, 379], [441, 1, 9480], [440, -3, 401], [432, -1, 405],
             [433, 1, 367], [443, -2, 401], [419, -11, 407], [432, 5, 405],
             [425, 1, 1500074], [443, -2, 401], [416, 2, 407], [445, 1, 393],
             [455, 0, 53503], [426, 1, 379], [440, -2, 380], [432, -1, 405], [440, 1, 9480],
             [443, -2, 401], [432, -1, 405], [440, -1, 401], [432, -1, 405], [426, 2, 379],
             [441, 1, 9480], [441, -3, 401], [419, -12, 407], [424, 6, 372], [447, 5, 365],
             [426, 3, 379], [433, 2, 367], [440, 1, 9480], [440, 1, 393], [455, 32, 373],
             [446, 1, 402], [440, -2, 380], [424, -3, 372], [455, -4, 375], [432, 3, 387],
             [445, 2, 393], [437, 1, 370], [454, -4, 373], [445, -1, 93815], [426, 2, 379],
             [433, 1, 367], [440, -4, 380], [420, -1, 407], [465, -2, 407], [445, 1, 393],
             [455, -4, 63716], [441, 1, 9480], [441, 1, 393], [458, -3, 373], [445, 1, 393],
             [458, -2, 373], [452, 1, 370], [454, -2, 411], [443, -1, 380], [432, -1, 405],
             [426, 2, 379], [441, 1, 9480], [441, -3, 401], [416, 3, 407], [415, 2, 407],
             [445, 1, 393], [455, 0, 53503], [432, -1, 405], [433, 1, 367], [440, -2, 401],
             [419, -4, 407], [465, 1, 407], [459, 0, 53503], [426, -1, 393], [461, 2, 380],
             [445, 1, 393], [458, -3, 373], [446, 1, 402], [443, 2, 393], [452, 1, 370],
             [454, -4, 373], [419, -9, 407], [415, -10, 407]],
            dtype='int32')
        self.spacy_doc.from_array(cols, values)

    def test_words(self):
        expected = [
            'Two', 'weeks', 'ago', ',', 'I', 'was', 'in', 'Kuwait', 'participating',
            'in', 'an', 'I.M.F.', 'seminar', 'for', 'Arab', 'educators', '.', 'For',
            '30', 'minutes', ',', 'we', 'discussed', 'the', 'impact']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False)][:25]
        self.assertEqual(observed, expected)

    def test_words_filter(self):
        expected = [
            'weeks', 'ago', 'Kuwait', 'participating', 'I.M.F.', 'seminar', 'Arab',
            'educators', 'minutes', 'discussed', 'impact', 'technology', 'trends',
            'education', 'Middle', 'East', 'Egyptian', 'education', 'official',
            'raised', 'hand', 'asked', 'ask', 'personal', 'question']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=True, filter_punct=True, filter_nums=True)][:25]
        self.assertEqual(observed, expected)

    def test_words_good_tags(self):
        expected = [
            'weeks', 'seminar', 'educators', 'minutes', 'impact', 'technology',
            'trends', 'education', 'education', 'official', 'hand', 'question',
            'mosques', 'sorrow', 'what', 'kids']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False,
            include_pos={'NOUN'})][:25]
        self.assertEqual(observed, expected)

    def test_words_min_freq(self):
        expected = [
            ',', 'I', 'was', 'in', 'in', 'an', 'for', '.', 'For', ',', 'we', 'the',
            'education', 'in', 'the', '.', 'And', 'an', 'education', 'and', 'asked',
            'he', 'ask', '"', 'I']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False,
            min_freq=2)][:25]
        self.assertEqual(observed, expected)

    def test_ngrams_less_than_1(self):
        with self.assertRaises(ValueError):
            list(extract.ngrams(self.spacy_doc, 0))

    def test_ngrams_1(self):
        expected = [
            'Two', 'weeks', 'ago', ',', 'I', 'was', 'in', 'Kuwait', 'participating',
            'in', 'an', 'I.M.F.', 'seminar', 'for', 'Arab', 'educators', '.', 'For',
            '30', 'minutes', ',', 'we', 'discussed', 'the', 'impact']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 1, filter_stops=False, filter_punct=False, filter_nums=False)][:25]
        self.assertEqual(observed, expected)

    def test_ngrams_2(self):
        expected = [
            'Two weeks', 'weeks ago', 'ago,', ', I', 'I was', 'was in', 'in Kuwait',
            'Kuwait participating', 'participating in', 'in an', 'an I.M.F.', 'I.M.F. seminar',
            'seminar for', 'for Arab', 'Arab educators', 'educators.', '. For', 'For 30',
            '30 minutes', 'minutes,', ', we', 'we discussed', 'discussed the', 'the impact',
            'impact of']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False)][:25]
        self.assertEqual(observed, expected)

    def test_ngrams_filter(self):
        expected = [
            'weeks ago', 'Kuwait participating', 'I.M.F. seminar', 'Arab educators',
            'technology trends', 'Middle East', 'Egyptian education', 'education official',
            'official raised', 'personal question', 'heard Donald', 'Donald Trump',
            'close mosques', 'United States', 'great sorrow']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=True, filter_punct=True, filter_nums=True)]
        self.assertEqual(observed, expected)

    def test_ngrams_min_freq(self):
        expected = ['in the', 'in the']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False,
            min_freq=2)]
        self.assertEqual(observed, expected)

    def test_ngrams_good_tag(self):
        expected = ['technology trends', 'education official']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False,
            include_pos={'NOUN'})]
        self.assertEqual(observed, expected)

    def test_named_entities(self):
        expected = [
            'Two weeks ago', 'Kuwait', 'Arab', '30 minutes', 'Middle East', 'Egyptian',
            'Donald Trump', 'United States']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=True)]
        self.assertEqual(observed, expected)

    def test_named_entities_good(self):
        expected = ['Kuwait', 'Donald Trump', 'United States']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, include_types={'PERSON', 'GPE'}, drop_determiners=True)]
        self.assertEqual(observed, expected)

    def test_named_entities_min_freq(self):
        expected = []
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=True, min_freq=2)]
        self.assertEqual(observed, expected)

    def test_named_entities_determiner(self):
        expected = ['the Middle East', 'the United States']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=False) if ent[0].pos_ == 'DET']
        self.assertEqual(observed, expected)

    @unittest.skip('waiting to hear back from spaCy, see issue #365')
    def test_noun_chunks(self):
        expected = [
            'I', 'Kuwait', 'I.M.F. seminar', 'Arab educators', '30 minutes', 'we',
            'impact', 'technology trends', 'education', 'Middle East', 'Egyptian education official',
            'his hand', 'he', 'personal question', 'I', 'Donald Trump', 'we', 'mosques',
            'United States', 'he', 'great sorrow', 'what', 'we', 'our kids']
        observed = [nc.text for nc in extract.noun_chunks(
            self.spacy_doc, drop_determiners=True)]
        self.assertEqual(observed, expected)

    @unittest.skip('waiting to hear back from spaCy, see issue #365')
    def test_noun_chunks_determiner(self):
        expected = [
            'I', 'Kuwait', 'an I.M.F. seminar', 'Arab educators', '30 minutes', 'we',
            'the impact', 'technology trends', 'education', 'the Middle East',
            'an Egyptian education official', 'his hand', 'he', 'a personal question',
            'I', 'Donald Trump', 'we', 'mosques', 'the United States', 'he', 'great sorrow',
            'what', 'we', 'our kids']
        observed = [nc.text for nc in extract.noun_chunks(
            self.spacy_doc, drop_determiners=False)]
        self.assertEqual(observed, expected)

    @unittest.skip('waiting to hear back from spaCy, see issue #365')
    def test_noun_chunks_min_freq(self):
        expected = ['I', 'we', 'he', 'I', 'we', 'he', 'we']
        observed = [nc.text for nc in extract.noun_chunks(
            self.spacy_doc, drop_determiners=True, min_freq=2)]
        self.assertEqual(observed, expected)

    def test_pos_regex_matches(self):
        expected = [
            'Two weeks', 'Kuwait', 'an I.M.F. seminar', 'Arab educators',
            '30 minutes', 'the impact', 'technology trends', 'education',
            'the Middle East', 'an Egyptian education official', 'his hand',
            'a personal question', 'Donald Trump', 'mosques',
            'the United States', 'great sorrow', 'that what', 'our kids']
        observed = [span.text for span in extract.pos_regex_matches(
            self.spacy_doc, constants.POS_REGEX_PATTERNS['en']['NP'])]
        self.assertEqual(observed, expected)

    def test_subject_verb_object_triples(self):
        expected = [
            'we, discussed, impact', 'education official, raised, hand', 'he, could ask, me',
            'he, could ask, question', 'we, need, to close']
        observed = [', '.join(item.text for item in triple) for triple in
                    extract.subject_verb_object_triples(self.spacy_doc)]
        self.assertEqual(observed, expected)

    def test_acronyms_and_definitions(self):
        expected = {'I.M.F.': ''}
        observed = extract.acronyms_and_definitions(self.spacy_doc)
        self.assertEqual(observed, expected)

    def test_acronyms_and_definitions_known(self):
        expected = {'I.M.F.': 'International Monetary Fund'}
        observed = extract.acronyms_and_definitions(
            self.spacy_doc, known_acro_defs={'I.M.F.': 'International Monetary Fund'})
        self.assertEqual(observed, expected)

    @unittest.skip("direct quotation extraction needs to be improved; it fails here")
    def test_direct_quotations(self):
        expected = [
            'he, said, "I heard Donald Trump say we need to close mosques in the United States,"',
            'he, said, "Is that what we want our kids to learn?"']
        observed = [', '.join(item.text for item in triple) for triple in
                    extract.direct_quotations(self.spacy_doc)]
        self.assertEqual(observed, expected)
