# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

import numpy as np
from spacy import attrs

from textacy import constants, data, extract


class ExtractTestCase(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        spacy_lang = data.load_spacy('en_core_web_sm')
        text = """
            Two weeks ago, I was in Kuwait participating in an I.M.F. seminar for Arab educators. For 30 minutes, we discussed the impact of technology trends on education in the Middle East. And then an Egyptian education official raised his hand and asked if he could ask me a personal question: "I heard Donald Trump say we need to close mosques in the United States," he said with great sorrow. "Is that what we want our kids to learn?"
            """
        self.spacy_doc = spacy_lang(text.strip())
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP, attrs.ENT_TYPE]
        values = np.array(
            [[459, 1, 758136, 387], [477, 1, 424, 387], [481, 3, 396, 387],
             [450, 2, 441, 0], [479, 1, 425, 0], [489, 0, 512817, 0],
             [466, -1, 439, 0], [475, -1, 435, 381], [490, -3, 395, 0],
             [466, -1, 439, 0], [460, 2, 411, 0], [475, 1, 74185, 0],
             [474, -3, 435, 0], [466, -1, 439, 0], [467, 1, 398, 378],
             [477, -2, 435, 0], [453, -11, 441, 0], [466, 5, 439, 388],
             [459, 1, 758136, 388], [477, -2, 435, 388], [450, 2, 441, 0],
             [479, 1, 425, 0], [489, 0, 512817, 0], [460, 1, 411, 0],
             [474, -2, 412, 0], [466, -1, 439, 0], [474, 1, 74185, 0],
             [477, -2, 435, 0], [466, -1, 439, 0], [474, -1, 435, 0],
             [466, -1, 439, 0], [460, 2, 411, 382], [475, 1, 74185, 382],
             [475, -3, 435, 382], [453, -12, 441, 0], [458, 6, 403, 0],
             [481, 5, 396, 0], [460, 3, 411, 0], [467, 2, 398, 378],
             [474, 1, 74185, 0], [474, 1, 425, 0], [489, 32, 404, 0],
             [480, 1, 436, 0], [474, -2, 412, 0], [458, -3, 403, 0],
             [489, -4, 406, 0], [466, 3, 419, 0], [479, 2, 425, 0],
             [471, 1, 401, 0], [488, -4, 395, 0], [479, -1, 758134, 0],
             [460, 2, 411, 0], [467, 1, 398, 0], [474, -4, 412, 0],
             [454, -1, 441, 0], [499, 2, 441, 0], [479, 1, 425, 0],
             [489, -8, 404, 0], [475, 1, 74185, 377], [475, 1, 425, 377],
             [492, -3, 404, 0], [479, 1, 425, 0], [492, -2, 404, 0],
             [486, 1, 401, 0], [488, -2, 445, 0], [477, -1, 412, 0],
             [466, -1, 439, 0], [460, 2, 411, 381], [475, 1, 74185, 381],
             [475, -3, 435, 381], [450, 3, 441, 0], [449, 2, 441, 0],
             [479, 1, 425, 0], [489, 0, 512817, 0], [466, -1, 439, 0],
             [467, 1, 398, 0], [474, -2, 435, 0], [453, -4, 441, 0],
             [499, 1, 441, 0], [493, 0, 512817, 0], [460, -1, 425, 0],
             [495, 2, 412, 0], [479, 1, 425, 0], [492, -4, 404, 0],
             [480, 1, 436, 0], [477, -6, 412, 0], [486, 1, 401, 0],
             [488, -8, 445, 0], [453, -9, 441, 0], [449, -10, 441, 0]],
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
            ',', 'I', 'in', 'in', 'an', 'for', '.', 'For', ',', 'we',
            'the', 'education', 'in', 'the', '.', 'And', 'an',
            'education', 'and', 'he', '"', 'I', 'we', 'to', 'in']
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
            'Two weeks ago', 'Kuwait', 'Arab', 'For 30 minutes', 'Middle East', 'Egyptian',
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
