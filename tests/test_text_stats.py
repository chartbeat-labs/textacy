from __future__ import absolute_import, unicode_literals

import unittest

from textacy import text_stats
from textacy import Doc


class TextStatsTestCase(unittest.TestCase):
    def setUp(self):
        self.spacy_doc = Doc('This is an English-language document.')
        self.n_chars = 2855
        self.n_syllables = 857
        self.n_words = 441
        self.n_polysyllable_words = 104
        self.n_monosyllable_words = 187
        self.n_long_words = 205
        self.n_sents = 21

    def test_readability_stats(self):
        self.assertIsInstance(text_stats.readability_stats(self.spacy_doc), dict)

    def test_readability_stats_lang(self):
        self.spacy_doc.lang = 'es'
        self.assertRaises(
            NotImplementedError, text_stats.readability_stats, self.spacy_doc)

    def test_flesch_kincaid_grade_level(self):
        self.assertAlmostEqual(
            text_stats.flesch_kincaid_grade_level(self.n_syllables, self.n_words, self.n_sents),
            15.53, places=2)

    def test_flesch_readability_ease(self):
        self.assertAlmostEqual(
            text_stats.flesch_readability_ease(self.n_syllables, self.n_words, self.n_sents),
            21.12, places=2)

    def test_smog_index(self):
        self.assertAlmostEqual(
            text_stats.smog_index(self.n_polysyllable_words, self.n_sents, verbose=False),
            15.84, places=2)

    def test_gunning_fog_index(self):
        self.assertAlmostEqual(
            text_stats.gunning_fog_index(self.n_words, self.n_polysyllable_words, self.n_sents),
            17.83, places=2)

    def test_coleman_liau_index(self):
        self.assertAlmostEqual(
            text_stats.coleman_liau_index(self.n_chars, self.n_words, self.n_sents),
            20.86, places=2)

    def test_automated_readability_index(self):
        self.assertAlmostEqual(
            text_stats.automated_readability_index(self.n_chars, self.n_words, self.n_sents),
            19.56, places=2)

    def test_wiener_sachtextformel_1(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(self.n_words, self.n_sents, self.n_polysyllable_words,
                                             self.n_monosyllable_words, self.n_long_words, variant=1),
            11.84, places=2)

    def test_wiener_sachtextformel_2(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(self.n_words, self.n_sents, self.n_polysyllable_words,
                                             self.n_monosyllable_words, self.n_long_words, variant=2),
            11.87, places=2)

    def test_wiener_sachtextformel_3(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(self.n_words, self.n_sents, self.n_polysyllable_words,
                                             self.n_monosyllable_words, self.n_long_words, variant=3),
            9.87, places=2)

    def test_wiener_sachtextformel_4(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(self.n_words, self.n_sents, self.n_polysyllable_words,
                                             self.n_monosyllable_words, self.n_long_words, variant=4),
            10.36, places=2)

    def test_gulpease_index(self):
        self.assertAlmostEqual(
            text_stats.gulpease_index(self.n_chars, self.n_words, self.n_sents),
            38.55, places=2)
