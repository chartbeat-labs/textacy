from __future__ import absolute_import, unicode_literals

from operator import itemgetter
import unittest

import numpy as np
from spacy import attrs

import textacy


class ReadmeTestCase(unittest.TestCase):

    def setUp(self):
        self.text = """
        Mr. President, I ask to have printed in the Record copies of some of the finalist essays written by Vermont High School students as part of the sixth annual ``What is the State of the Union'' essay contest conducted by my office. These finalists were selected from nearly 800 entries. The material follows: The United States of America is an amazing nation that continues to lead the world through the complex geopolitical problems that we are faced with today. As a strong economic and political world leader, we have become the role model for developing nations attempting to give their people the same freedoms and opportunities that Americans have become so accustomed to. This is why it is so important to work harder than we ever have before to better ourselves as a nation, because what we change will set a precedent of improvement around the world and inspire change. The biggest problem in the U.S. is the incarceration system. It has been broken for decades, and there has been no legitimate attempt to fix it. Over the past thirty years, there has been a 500% increase in incarceration rates, resulting in the U.S. leading the world in number of prisoners with 2.2 million people currently incarcerated. Especially in this example, it is important to humanize these statistics. These are 2.2 million people, who now because of their conviction will find it much harder to be truly integrated back in their communities, due to the struggles of finding a job with a record, and the fact that they often do not qualify for social welfare. The incarceration system is also bankrupting both the state and federal government. It currently is the third highest state expenditure, behind health care and education. Fortunately, we as a nation have the opportunity to fix the incarceration system. First, we need to get rid of mandatory minimum sentences. Judges from across the nation have said for decades that they do not like mandatory minimums, that they do not work, and that they are unconstitutional. Mandatory minimum sentences, coupled with racially biased laws concerning drug possession is the reason why we see the ratio of African American males to white males over 10:1. This leads to the second action we must take; we must end the war on drugs. It has proven to be a failed experiment that has reopened many racial wounds in our nation. The war on drugs also put addicts behind bars, rather than treating addiction like the problem it actually is; a mental health issue.
        """
        self.spacy_lang = textacy.data.load_spacy('en_core_web_sm')
        self.doc = textacy.Doc(self.text.strip(), lang=self.spacy_lang)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[485, 1, 0], [475, 1, 74185], [475, 3, 758135], [450, -1, 441],
             [479, 1, 425], [492, 0, 512817], [486, 2, 401], [488, 1, 401],
             [491, -3, 445], [466, -1, 439], [460, 2, 411], [475, 1, 74185],
             [477, -3, 435], [466, -1, 439], [460, -1, 435], [466, -1, 439],
             [460, 2, 411], [467, 1, 398], [477, -3, 435], [491, -1, 758131],
             [466, -1, 397], [475, 2, 74185], [475, 1, 74185], [475, 1, 74185],
             [477, -4, 435], [466, -6, 439], [474, -1, 435], [466, -1, 439],
             [460, 2, 411], [467, 1, 398], [467, 11, 398], [499, -1, 441],
             [495, 1, 400], [493, 8, 408], [460, 1, 411], [475, -2, 425],
             [466, -1, 439], [460, 1, 411], [475, -2, 435], [449, -1, 441],
             [474, 1, 74185], [474, -14, 435], [491, -1, 758131], [466, -1, 397],
             [480, 1, 436], [474, -2, 435], [453, -41, 441], [460, 1, 411],
             [477, 2, 426], [489, 1, 402], [491, 0, 512817], [466, -1, 439],
             [481, 1, 396], [459, 1, 758136], [477, -3, 435], [453, -5, 441],
             [460, 1, 411], [474, 1, 425], [493, 0, 512817], [454, -1, 441],
             [460, 2, 411], [475, 1, 74185], [475, 3, 425], [466, -1, 439],
             [475, -1, 435], [493, -7, 404], [460, 2, 411], [467, 1, 398],
             [474, -3, 400], [494, 1, 425], [493, -2, 758141], [486, 1, 401],
             [488, -2, 445], [460, 1, 411], [474, -2, 412], [466, -3, 439],
             [460, 3, 411], [467, 2, 398], [467, 1, 398], [477, -4, 435],
             [494, 3, 419], [479, 2, 426], [492, 1, 402], [491, -4, 758141],
             [466, -1, 439], [474, -1, 435], [453, -28, 441], [466, 11, 439],
             [460, 6, 411], [467, 5, 398], [467, 4, 398], [458, -1, 403],
             [467, -2, 406], [474, 1, 74185], [474, -7, 435], [450, 3, 441],
             [479, 2, 425], [492, 1, 401], [491, 0, 512817], [460, 2, 411],
             [474, 1, 74185], [474, -3, 400], [466, -1, 439], [490, -1, 434],
             [477, -1, 412], [490, -1, 758131], [486, 1, 401], [488, -2, 445],
             [480, 1, 436], [477, -2, 758134], [460, 2, 411], [467, 1, 398],
             [477, -5, 412], [458, -1, 403], [477, -2, 406], [494, 3, 419],
             [476, 2, 425], [492, 1, 401], [491, -6, 758141], [481, 1, 396],
             [467, -2, 394], [466, -1, 439], [453, -24, 441], [460, 1, 425],
             [493, 0, 512817], [497, 2, 396], [479, 1, 425], [493, -3, 404],
             [481, 1, 396], [467, -2, 394], [486, 1, 401], [488, -2, 445],
             [482, -1, 396], [466, 3, 419], [479, 2, 425], [481, 1, 396],
             [492, -4, 395], [466, -1, 439], [486, 1, 401], [482, -2, 445],
             [479, -1, 412], [466, -2, 439], [460, 1, 411], [474, -2, 435],
             [450, -8, 441], [466, 5, 419], [495, 2, 412], [479, 1, 425],
             [492, 2, 408], [471, 1, 401], [488, 9, 395], [460, 1, 411],
             [474, -2, 412], [466, -1, 439], [474, -1, 435], [466, -1, 439],
             [460, 1, 411], [474, -2, 435], [458, -6, 403], [488, -35, 406],
             [474, -1, 412], [453, -37, 441], [460, 2, 411], [469, 1, 398],
             [474, 4, 425], [466, -1, 439], [460, 1, 411], [475, -2, 435],
             [493, 0, 512817], [460, 2, 411], [474, 1, 74185], [474, -3, 400],
             [453, -4, 441], [479, 3, 426], [493, 2, 401], [491, 1, 402],
             [491, 0, 512817], [466, -1, 439], [477, -1, 435], [450, -3, 441],
             [458, -4, 403], [461, 2, 413], [493, 1, 401], [491, -7, 406],
             [460, 2, 411], [467, 1, 398], [474, -3, 400], [486, 1, 401],
             [488, -2, 758131], [479, -1, 412], [453, -7, 441], [466, 8, 439],
             [460, 3, 411], [467, 2, 398], [459, 1, 758136], [477, -4, 435],
             [450, 3, 441], [461, 2, 413], [493, 1, 401], [491, 0, 512817],
             [460, 2, 411], [459, 1, 758136], [474, -3, 400], [466, -1, 439],
             [474, 1, 74185], [477, -2, 435], [450, -1, 441], [490, -2, 758131],
             [466, -1, 439], [460, 1, 411], [475, -2, 435], [490, -4, 406],
             [460, 1, 411], [474, -2, 412], [466, -3, 439], [474, -1, 435],
             [466, -1, 439], [477, -1, 435], [466, -1, 439], [459, 1, 74185],
             [459, 1, 758136], [477, -3, 435], [481, 1, 396], [491, -8, 758131],
             [453, -25, 441], [481, 6, 396], [466, -1, 439], [460, 1, 411],
             [474, -2, 435], [450, 2, 441], [479, 1, 425], [493, 0, 512817],
             [467, -1, 394], [486, 1, 401], [488, -3, 445], [460, 1, 411],
             [477, -2, 412], [453, -6, 441], [460, 1, 425], [492, 0, 512817],
             [459, 1, 74185], [459, 1, 758136], [477, -3, 400], [450, -1, 441],
             [495, 7, 425], [481, 6, 396], [466, 5, 439], [466, -1, 434],
             [480, 1, 436], [474, 2, 425], [471, 1, 401], [488, -9, 758141],
             [479, 2, 425], [481, 1, 396], [482, -3, 404], [486, 3, 401],
             [488, 2, 402], [481, 1, 396], [491, -4, 445], [481, -1, 396],
             [466, -1, 439], [480, 1, 436], [477, -2, 435], [450, -1, 441],
             [466, -6, 439], [466, -1, 434], [460, 1, 411], [477, -3, 435],
             [466, -1, 439], [490, -1, 434], [460, 1, 411], [474, -2, 412],
             [466, -1, 439], [460, 1, 411], [474, -2, 435], [450, -1, 441],
             [458, -2, 403], [460, 1, 411], [474, -4, 406], [466, 5, 419],
             [479, 4, 425], [481, 3, 396], [492, 2, 401], [481, 1, 421],
             [488, -6, 758131], [466, -1, 439], [467, 1, 398], [474, -2, 435],
             [453, -49, 441], [460, 2, 411], [474, 1, 74185], [474, 3, 425],
             [493, 2, 401], [481, 1, 396], [490, 0, 512817], [458, 5, 438],
             [460, 4, 411], [474, 3, 422], [458, -1, 403], [467, -2, 406],
             [474, -6, 412], [453, -7, 441], [479, 2, 425], [481, 1, 396],
             [493, 0, 512817], [460, 4, 411], [467, 3, 398], [469, 2, 398],
             [474, 1, 74185], [474, -5, 400], [450, -1, 441], [466, -2, 439],
             [474, 1, 74185], [474, -2, 435], [458, -1, 403], [474, -2, 406],
             [453, -12, 441], [481, 6, 396], [450, 5, 441], [479, 4, 425],
             [466, -1, 439], [460, 1, 411], [474, -2, 435], [492, 0, 512817],
             [460, 1, 411], [474, -2, 412], [486, 1, 401], [488, -2, 758131],
             [460, 2, 411], [474, 1, 74185], [474, -3, 412], [453, -8, 441],
             [481, 3, 396], [450, 2, 441], [479, 1, 425], [492, 0, 512817],
             [486, 1, 401], [488, 1, 402], [491, -3, 445], [466, -1, 439],
             [467, 2, 398], [467, 1, 398], [477, -3, 435], [453, -8, 441],
             [477, 6, 425], [466, -1, 439], [466, -1, 439], [460, 1, 411],
             [474, -2, 435], [492, 1, 401], [491, 0, 512817], [466, -1, 439],
             [477, -1, 435], [494, 4, 419], [479, 3, 425], [492, 2, 401],
             [481, 1, 421], [488, -5, 758141], [467, 1, 398], [477, -2, 412],
             [450, -8, 441], [466, 4, 419], [479, 3, 425], [492, 2, 401],
             [481, 1, 421], [488, -13, 758131], [450, -16, 441], [458, -17, 403],
             [466, 2, 419], [479, 1, 425], [492, -20, 406], [467, -1, 394],
             [453, -22, 441], [467, 2, 398], [467, 1, 398], [477, 10, 425],
             [450, -1, 441], [491, -2, 758131], [466, -1, 439], [481, 1, 396],
             [467, 1, 398], [477, -3, 435], [490, -1, 758131], [474, 1, 74185],
             [474, -2, 412], [493, 0, 512817], [460, 1, 411], [474, -2, 425],
             [497, 2, 396], [479, 1, 425], [492, -3, 758141], [460, 1, 411],
             [474, -2, 412], [466, -1, 439], [467, 2, 398], [467, 1, 398],
             [477, -3, 435], [466, -5, 439], [467, 1, 398], [477, -2, 435],
             [466, -1, 439], [459, -1, 435], [453, -17, 441], [460, 1, 425],
             [493, 11, 404], [466, -1, 439], [460, 2, 411], [467, 1, 398],
             [474, -3, 435], [479, 2, 425], [471, 1, 401], [488, -3, 758141],
             [454, 3, 441], [479, 2, 425], [471, 1, 401], [488, 0, 512817],
             [460, 1, 411], [474, -2, 412], [466, -1, 439], [477, -1, 435],
             [453, -5, 441], [479, 2, 425], [493, 1, 401], [491, 0, 512817],
             [486, 1, 401], [488, -2, 445], [460, 2, 411], [491, 1, 398],
             [474, -3, 400], [494, 2, 425], [493, 1, 401], [491, -3, 758141],
             [467, 2, 398], [467, 1, 398], [477, -3, 412], [466, -4, 439],
             [480, 1, 436], [474, -2, 435], [453, -15, 441], [460, 1, 411],
             [474, 4, 425], [466, -1, 439], [477, -1, 435], [481, 1, 396],
             [491, 0, 512817], [477, -1, 412], [466, -2, 439], [477, -1, 435],
             [450, -1, 441], [481, 1, 396], [466, -6, 403], [490, -1, 434],
             [474, -1, 412], [466, -1, 439], [460, 1, 411], [474, -2, 435],
             [479, 2, 425], [481, 1, 396], [493, -3, 758141], [454, -15, 441],
             [460, 3, 411], [467, 2, 398], [474, 1, 74185], [474, -19, 406],
             [453, -20, 441], [485, -1, 0]],
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
                      textacy.extract.ngrams(self.doc, 2, filter_stops=True, filter_punct=True, filter_nums=False)][:15]
        expected_1 = ['Mr. President', 'Record copies', 'finalist essays',
                      'essays written', 'Vermont High', 'High School',
                      'School students', 'sixth annual', 'annual ``',
                      'essay contest', 'contest conducted', 'nearly 800',
                      '800 entries', 'material follows', 'United States']
        observed_2 = [ng.text for ng in
                      textacy.extract.ngrams(self.doc, 3, filter_stops=True, filter_punct=True, min_freq=2)]
        expected_2 = ['2.2 million people', '2.2 million people',
                      'mandatory minimum sentences', 'Mandatory minimum sentences',
                      'war on drugs', 'war on drugs']
        observed_3 = [ne.text for ne in
                      textacy.extract.named_entities(self.doc, drop_determiners=True, exclude_types='numeric')]
        expected_3 = ['Vermont High School',
                      'United States of America', 'Americans', 'U.S.', 'U.S.',
                      'African', 'American']
        observed_4 = [match.text for match in
                      textacy.extract.pos_regex_matches(self.doc, textacy.constants.POS_REGEX_PATTERNS['en']['NP'])][-10:]
        expected_4 = ['experiment', 'many racial wounds', 'our nation',
                      'The war', 'drugs', 'addicts', 'bars', 'addiction',
                      'the problem', 'a mental health issue']
        observed_5 = textacy.keyterms.textrank(self.doc, n_keyterms=5)
        expected_5 = [('nation', 0.039171899480447796),
                      ('world', 0.027185180109043087),
                      ('problem', 0.021632938680995067),
                      ('state', 0.021117306518205543),
                      ('people', 0.020564798929618617)]
        self.assertEqual(observed_1, expected_1)
        self.assertEqual(observed_2, expected_2)
        self.assertEqual(observed_3, expected_3)
        self.assertEqual(observed_4, expected_4)
        for o, e in zip(observed_5, expected_5):
            self.assertEqual(o[0], e[0])
            self.assertAlmostEqual(o[1], e[1], places=4)

    def test_readability_stats(self):
        ts = textacy.text_stats.TextStats(self.doc)
        observed_1 = ts.basic_counts
        observed_2 = ts.readability_stats
        expected_1 = {
            'n_chars': 2027,
            'n_long_words': 97,
            'n_monosyllable_words': 287,
            'n_polysyllable_words': 57,
            'n_sents': 19,
            'n_syllables': 648,
            'n_unique_words': 228,
            'n_words': 424}
        expected_2 = {
            'automated_readability_index': 12.244805114200595,
            'coleman_liau_index': 10.982921606132077,
            'flesch_kincaid_grade_level': 11.147120158887784,
            'flesch_readability_ease': 54.89013406156903,
            'gulpease_index': 54.63679245283019,
            'gunning_fog_index': 14.303674280039722,
            'lix': 45.193147964250244,
            'smog_index': 13.023866798666859,
            'wiener_sachtextformel': 6.211270754716981}
        for key in expected_1:
            self.assertEqual(observed_1[key], expected_1[key])
        for key in expected_2:
            self.assertAlmostEqual(observed_2[key], expected_2[key], places=2)

    def test_term_counting(self):
        observed_1 = self.doc.count('nation')
        expected_1 = 5
        bot = self.doc.to_bag_of_terms(
            ngrams=1, normalize='lemma', as_strings=True)
        # sort by term ascending, then count descending
        observed_2 = sorted(bot.items(), key=itemgetter(1, 0), reverse=True)[:10]
        expected_2 = [
            ('nation', 6), ('world', 4), ('u.s.', 4), ('incarceration', 4),
            ('decade', 4), ('state', 3), ('record', 3), ('problem', 3),
            ('people', 3), ('minimum', 3)]
        self.assertEqual(observed_1, expected_1)
        self.assertEqual(observed_2, expected_2)
