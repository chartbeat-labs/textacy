from __future__ import absolute_import, unicode_literals

import unittest

import numpy as np
from spacy import attrs

import textacy

class ReadmeTestCase(unittest.TestCase):

    def setUp(self):
        self.text = "Mr. President, I ask to have printed in the Record copies of some of the finalist essays written by Vermont High School students as part of the sixth annual ``What is the State of the Union'' essay contest conducted by my office. These finalists were selected from nearly 800 entries. The material follows: The United States of America is an amazing nation that continues to lead the world through the complex geopolitical problems that we are faced with today. As a strong economic and political world leader, we have become the role model for developing nations attempting to give their people the same freedoms and opportunities that Americans have become so accustomed to. This is why it is so important to work harder than we ever have before to better ourselves as a nation, because what we change will set a precedent of improvement around the world and inspire change. The biggest problem in the U.S. is the incarceration system. It has been broken for decades, and there has been no legitimate attempt to fix it. Over the past thirty years, there has been a 500% increase in incarceration rates, resulting in the U.S. leading the world in number of prisoners with 2.2 million people currently incarcerated. Especially in this example, it is important to humanize these statistics. These are 2.2 million people, who now because of their conviction will find it much harder to be truly integrated back in their communities, due to the struggles of finding a job with a record, and the fact that they often do not qualify for social welfare. The incarceration system is also bankrupting both the state and federal government. It currently is the third highest state expenditure, behind health care and education. Fortunately, we as a nation have the opportunity to fix the incarceration system. First, we need to get rid of mandatory minimum sentences. Judges from across the nation have said for decades that they do not like mandatory minimums, that they do not work, and that they are unconstitutional. Mandatory minimum sentences, coupled with racially biased laws concerning drug possession is the reason why we see the ratio of African American males to white males over 10:1. This leads to the second action we must take; we must end the war on drugs. It has proven to be a failed experiment that has reopened many racial wounds in our nation. The war on drugs also put addicts behind bars, rather than treating addiction like the problem it actually is; a mental health issue."
        self.doc = textacy.TextDoc(self.text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[441, 1, 9480], [441, 3, 392], [416, 2, 407], [445, 1, 393],
             [458, 0, 53503], [452, 2, 370], [454, 1, 370], [457, -3, 411],
             [432, -1, 405], [426, 2, 379], [441, 1, 9480], [443, -3, 401],
             [432, -1, 405], [426, -1, 401], [432, -1, 405], [426, 2, 379],
             [433, 1, 367], [443, -3, 401], [457, -1, 63716], [432, -1, 366],
             [441, 2, 9480], [441, 1, 9480], [441, 1, 9480], [443, -4, 401],
             [432, -6, 405], [440, -1, 401], [432, -1, 405], [426, 5, 379],
             [433, 4, 393], [433, 3, 367], [465, 2, 407], [461, 1, 393],
             [459, -28, 373], [426, 1, 379], [441, -2, 369], [432, -1, 405],
             [426, 1, 379], [441, -2, 401], [415, -6, 407], [440, 1, 9480],
             [440, 0, 53503], [457, -1, 63716], [432, -1, 366], [446, 1, 402],
             [440, -2, 401], [419, -5, 407], [426, 1, 379], [443, 2, 394],
             [455, 1, 371], [457, 0, 53503], [432, -1, 405], [447, 1, 365],
             [425, 1, 1500074], [443, -3, 401], [419, -5, 407], [426, 1, 379],
             [440, 1, 393], [459, 7, 373], [420, 6, 407], [426, 2, 379],
             [441, 1, 9480], [441, 3, 393], [432, -1, 405], [441, -1, 401],
             [459, 0, 53503], [426, 2, 379], [433, 1, 367], [440, -3, 369],
             [460, 1, 393], [459, -2, 1500076], [452, 1, 370], [454, -2, 411],
             [426, 1, 379], [440, -2, 380], [432, -3, 405], [426, 3, 379],
             [433, 2, 367], [433, 1, 367], [443, -4, 401], [460, 3, 380],
             [445, 2, 394], [458, 1, 371], [457, -4, 1500076], [432, -1, 405],
             [440, -1, 401], [419, -21, 407], [432, 11, 405], [426, 6, 379],
             [433, 5, 367], [433, 3, 367], [424, -1, 372], [433, -2, 375],
             [440, 1, 9480], [440, -7, 401], [416, 3, 407], [445, 2, 393],
             [458, 1, 370], [457, 0, 53503], [426, 2, 379], [440, 1, 9480],
             [440, -3, 369], [432, -1, 405], [456, 1, 367], [443, 1, 393],
             [456, -3, 400], [452, 1, 370], [454, -2, 411], [446, 1, 402],
             [443, -2, 93815], [426, 2, 379], [433, 1, 367], [443, -5, 380],
             [424, -1, 372], [443, -2, 375], [460, 3, 380], [442, 2, 393],
             [458, 1, 370], [457, -6, 1500076], [447, 1, 365], [433, -2, 363],
             [432, -1, 405], [419, -24, 407], [426, 1, 393], [459, 0, 53503],
             [463, 2, 365], [445, 1, 393], [459, -3, 364], [447, 1, 365],
             [433, -2, 363], [452, 1, 370], [454, -2, 411], [448, -1, 365],
             [432, 3, 387], [445, 2, 393], [447, 1, 365], [458, -4, 364],
             [432, 2, 387], [452, 1, 370], [447, -8, 364], [445, -1, 380],
             [432, -2, 405], [426, 1, 379], [440, -2, 401], [416, -13, 407],
             [432, 5, 387], [461, 2, 380], [445, 1, 393], [458, 2, 376],
             [437, 1, 370], [454, -19, 364], [426, 1, 379], [440, -2, 380],
             [432, -1, 405], [440, -1, 401], [432, -1, 405], [426, 1, 379],
             [440, -2, 401], [424, -8, 372], [440, -9, 375], [440, -1, 380],
             [419, -37, 407], [426, 2, 379], [435, 1, 367], [440, 4, 393],
             [432, -1, 405], [426, 1, 379], [441, -2, 401], [459, 0, 53503],
             [426, 2, 379], [440, 1, 9480], [440, -3, 369], [419, -4, 407],
             [445, 3, 394], [459, 2, 370], [457, 1, 371], [457, 0, 53503],
             [432, -1, 405], [443, -1, 401], [416, -3, 407], [424, -4, 372],
             [427, 2, 381], [459, 1, 370], [457, -7, 375], [426, 2, 379],
             [433, 1, 367], [440, -3, 369], [452, 1, 370], [454, -2, 63716],
             [445, -1, 380], [419, -7, 407], [432, 8, 405], [426, 3, 379],
             [433, 2, 367], [425, 1, 1500074], [443, -4, 401], [416, 3, 407],
             [427, 2, 381], [459, 1, 370], [457, 0, 53503], [426, 3, 379],
             [425, 1, 1500074], [440, 1, 9480], [440, -4, 369], [432, -1, 405],
             [440, 1, 9480], [443, -2, 401], [416, -8, 407], [456, -9, 364],
             [432, -1, 405], [426, 1, 379], [441, 1, 393], [456, -3, 400],
             [426, 1, 379], [440, -2, 380], [432, -3, 405], [440, -1, 401],
             [432, -1, 405], [443, -1, 401], [432, 5, 387], [425, 1, 9480],
             [425, 1, 1500074], [443, 2, 393], [447, 1, 365], [457, -12, 364],
             [419, -26, 407], [447, 1, 365], [432, 5, 405], [426, 1, 379],
             [440, -2, 401], [416, 2, 407], [445, 1, 393], [459, 0, 53503],
             [433, -1, 363], [452, 1, 370], [454, -3, 411], [426, 1, 379],
             [443, -2, 380], [419, -6, 407], [426, 1, 393], [458, 0, 53503],
             [425, 1, 9480], [425, 1, 1500074], [443, -3, 369], [416, -1, 407],
             [461, 2, 393], [447, 1, 365], [432, -7, 405], [432, -1, 400],
             [446, 1, 402], [440, -3, 401], [437, 1, 370], [454, -12, 375],
             [445, 2, 393], [447, 1, 365], [448, -3, 373], [452, 1, 370],
             [454, 2, 371], [447, 1, 365], [457, -7, 373], [447, -1, 365],
             [432, -1, 405], [446, 1, 402], [443, -2, 401], [416, -5, 407],
             [432, -6, 405], [432, -1, 400], [426, 1, 379], [443, -3, 401],
             [432, -1, 405], [456, -1, 400], [426, 1, 379], [440, -2, 380],
             [432, -1, 405], [426, 1, 379], [440, -2, 401], [416, -17, 407],
             [424, -18, 372], [426, 1, 379], [440, -20, 375], [432, 5, 387],
             [445, 4, 393], [447, 3, 365], [458, 2, 370], [447, 1, 389],
             [454, -6, 63716], [432, -1, 405], [433, 1, 367], [440, -2, 401],
             [419, -49, 407], [426, 2, 379], [440, 1, 9480], [440, 3, 393],
             [459, 2, 370], [447, 1, 365], [456, 0, 53503], [424, 2, 404],
             [426, 1, 379], [440, 3, 390], [424, -1, 372], [433, -2, 375],
             [440, -6, 380], [419, -7, 407], [445, 2, 393], [447, 1, 365],
             [459, 0, 53503], [426, 4, 379], [433, 3, 367], [435, 2, 367],
             [440, 1, 9480], [440, -5, 369], [416, -6, 407], [432, -7, 405],
             [440, 1, 9480], [440, -2, 401], [424, -1, 372], [440, -2, 375],
             [419, -12, 407], [447, 6, 365], [416, 5, 407], [445, 4, 393],
             [432, -1, 405], [426, 1, 379], [440, -2, 401], [458, 0, 53503],
             [426, 1, 379], [440, -2, 380], [452, 1, 370], [454, -2, 63716],
             [426, 2, 379], [440, 1, 9480], [440, -3, 380], [419, -8, 407],
             [447, 3, 365], [416, 2, 407], [445, 1, 393], [458, 0, 53503],
             [452, 1, 370], [454, -2, 411], [433, -1, 411], [432, -1, 405],
             [433, 2, 367], [433, 1, 367], [443, -3, 401], [419, -8, 407],
             [443, 6, 393], [432, -1, 405], [432, -1, 405], [426, 1, 379],
             [440, -2, 401], [458, 1, 370], [457, 0, 53503], [432, -1, 405],
             [443, -1, 401], [460, 4, 387], [445, 3, 393], [458, 2, 370],
             [447, 1, 389], [454, -7, 373], [433, 1, 367], [443, -2, 380],
             [416, -3, 407], [432, 4, 387], [445, 3, 393], [458, 2, 370],
             [447, 1, 389], [454, -8, 375], [416, -1, 407], [424, -2, 372],
             [432, 2, 387], [445, 1, 393], [458, -20, 373], [433, -1, 363],
             [419, -22, 407], [433, 2, 367], [433, 1, 367], [443, 10, 393],
             [416, -1, 407], [457, -2, 63716], [432, -1, 405], [447, 1, 365],
             [433, 1, 367], [443, -3, 401], [456, -1, 405], [440, 1, 9480],
             [440, -2, 401], [459, 0, 53503], [426, 1, 379], [440, -2, 369],
             [463, 2, 365], [445, 1, 393], [458, -3, 1500076], [426, 1, 379],
             [440, -2, 380], [432, -1, 405], [433, 2, 367], [433, 1, 367],
             [443, -3, 401], [432, -5, 405], [433, 1, 367], [443, -2, 401],
             [432, -8, 405], [425, -1, 401], [419, -17, 407], [426, 1, 393],
             [459, 11, 373], [432, -1, 405], [426, 2, 379], [433, 1, 367],
             [440, -3, 401], [445, 2, 393], [437, 1, 370], [454, -3, 1500076],
             [420, 3, 407], [445, 2, 393], [437, 1, 370], [454, 0, 53503],
             [426, 1, 379], [440, -2, 380], [432, -1, 405], [443, -1, 401],
             [419, -5, 407], [445, 2, 393], [459, 1, 370], [457, 0, 53503],
             [452, 1, 370], [454, -2, 411], [426, 2, 379], [457, 1, 367],
             [440, -3, 369], [460, 2, 393], [459, 1, 370], [457, -3, 1500076],
             [433, 2, 367], [433, 1, 367], [443, -3, 380], [432, -1, 405],
             [446, 1, 402], [440, -2, 401], [419, -15, 407], [426, 1, 379],
             [440, 4, 393], [432, -1, 405], [443, -1, 401], [447, 1, 365],
             [458, 0, 53503], [443, -1, 380], [432, -2, 405], [443, -1, 401],
             [416, -4, 407], [447, 1, 365], [432, -6, 372], [456, -7, 364],
             [440, -1, 380], [432, -2, 405], [426, 1, 379], [440, -2, 401],
             [445, 2, 393], [447, 1, 365], [459, -3, 1500076], [420, -1, 407],
             [426, 3, 379], [433, 1, 367], [440, 1, 9480], [440, -5, 369],
             [419, -20, 407]],
            dtype='int32')
        self.doc.spacy_doc.from_array(cols, values)

    def test_plaintext_functionality(self):
        expected_1 = 'mr president i ask to have printed in the record copies of some of the'
        observed_1 = textacy.preprocess_text(self.text, lowercase=True, no_punct=True)[:70]
        expected_2 = [('ed States of America is an amazing ',
                       'nation',
                       ' that continues to lead the world t'),
                      ('come the role model for developing ',
                       'nation',
                       's attempting to give their people t'),
                      ('ve before to better ourselves as a ',
                       'nation',
                       ', because what we change will set a'),
                      ('nd education. Fortunately, we as a ',
                       'nation',
                       ' have the opportunity to fix the in'),
                      (' sentences. Judges from across the ',
                       'nation',
                       ' have said for decades that they do'),
                      ('reopened many racial wounds in our ',
                       'nation',
                       '. The war on drugs also put addicts')]
        observed_2 = list(textacy.text_utils.keyword_in_context(
            self.text, 'nation', window_width=35, print_only=False))
        self.assertEqual(observed_1, expected_1)
        self.assertEqual(observed_2, expected_2)

    def test_extract_functionality(self):
        observed_1 = [ng.text for ng in
                      self.doc.ngrams(2, filter_stops=True, filter_punct=True, filter_nums=False)][:15]
        expected_1 = ['Mr. President', 'Record copies', 'finalist essays',
                      'essays written', 'Vermont High', 'High School',
                      'School students', 'sixth annual', 'annual ``',
                      'essay contest', 'contest conducted', 'nearly 800',
                      '800 entries', 'material follows', 'United States']
        observed_2 = [ng.text for ng in
                      self.doc.ngrams(3, filter_stops=True, filter_punct=True, min_freq=2)]
        expected_2 = ['lead the world', 'leading the world',
                      '2.2 million people', '2.2 million people',
                      'mandatory minimum sentences',
                      'Mandatory minimum sentences', 'war on drugs',
                      'war on drugs']
        observed_3 = [ne.text for ne in
                      self.doc.named_entities(drop_determiners=True, bad_ne_types='numeric')]
        expected_3 = ['Record', 'Vermont High School',
                      'United States of America', 'Americans', 'U.S.', 'U.S.',
                      'African American']
        observed_4 = [match.text for match in
                      self.doc.pos_regex_matches(textacy.regexes_etc.POS_REGEX_PATTERNS['en']['NP'])][-10:]
        expected_4 = ['experiment', 'many racial wounds', 'our nation',
                      'The war', 'drugs', 'addicts', 'bars', 'addiction',
                      'the problem', 'a mental health issue']
        observed_5 = self.doc.key_terms(algorithm='textrank', n=5)
        expected_5 = [('nation', 0.04315758994993049),
                      ('world', 0.030590559641614556),
                      ('incarceration', 0.029577233127175532),
                      ('problem', 0.02411902162606202),
                      ('people', 0.022631145896105508)]
        self.assertEqual(observed_1, expected_1)
        self.assertEqual(observed_2, expected_2)
        self.assertEqual(observed_3, expected_3)
        self.assertEqual(observed_4, expected_4)
        for o, e in zip(observed_5, expected_5):
            self.assertEqual(o[0], e[0])
            self.assertAlmostEqual(o[1], e[1], places=4)

    def test_readability(self):
        observed = self.doc.readability_stats
        expected = {'automated_readability_index': 11.67580188679245,
                    'coleman_liau_index': 10.89927271226415,
                    'flesch_kincaid_grade_level': 10.711962264150948,
                    'flesch_readability_ease': 56.022660377358505,
                    'gunning_fog_index': 13.857358490566037,
                    'n_chars': 2026,
                    'n_polysyllable_words': 57,
                    'n_sents': 20,
                    'n_syllables': 648,
                    'n_unique_words': 228,
                    'n_words': 424,
                    'smog_index': 12.773325707644965}
        for key in expected:
            self.assertAlmostEqual(observed[key], expected[key], places=4)

    def test_term_counting(self):
        observed_1 = self.doc.term_count('nation')
        expected_1 = 6
        bot = self.doc.as_bag_of_terms(weighting='tf', normalized=False,
                                       lemmatize='auto', ngram_range=(1, 1))
        observed_2 = sorted([(self.doc.spacy_stringstore[term_id], count)
                             for term_id, count in bot.most_common(n=10)],
                            key=lambda x: x[1], reverse=True)
        expected_2 = [('nation', 6), ('incarceration', 4), ('world', 4), ('lead', 3),
                      ('mandatory', 3), ('people', 3), ('minimum', 3), ('drug', 3),
                      ('problem', 3), ('male', 2)]
        self.assertEqual(observed_1, expected_1)
        self.assertEqual(observed_2, expected_2)
