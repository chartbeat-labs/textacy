# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from textacy import text_stats
from textacy import Doc

TEXT = """
Mr. Speaker, 480,000 Federal employees are working without pay, a form of involuntary servitude; 280,000 Federal employees are not working, and they will be paid. Virtually all of these workers have mortgages to pay, children to feed, and financial obligations to meet.
Mr. Speaker, what is happening to these workers is immoral, is wrong, and must be rectified immediately. Newt Gingrich and the Republican leadership must not continue to hold the House and the American people hostage while they push their disastrous 7-year balanced budget plan. The gentleman from Georgia, Mr. Gingrich, and the Republican leadership must join Senator Dole and the entire Senate and pass a continuing resolution now, now to reopen Government.
Mr. Speaker, that is what the American people want, that is what they need, and that is what this body must do.
"""


class TextStatsTestCase(unittest.TestCase):

    def setUp(self):
        self.doc = Doc(TEXT, lang='en')
        self.ts = text_stats.TextStats(self.doc)

    def test_n_sents(self):
        self.assertEqual(self.ts.n_sents, 6)

    def test_n_words(self):
        self.assertEqual(self.ts.n_words, 136)

    def test_n_chars(self):
        self.assertEqual(self.ts.n_chars, 685)

    def test_n_syllables(self):
        self.assertEqual(self.ts.n_syllables, 214)

    def test_n_unique_words(self):
        self.assertEqual(self.ts.n_unique_words, 80)

    def test_n_long_words(self):
        self.assertEqual(self.ts.n_long_words, 43)

    def test_n_monosyllable_words(self):
        self.assertEqual(self.ts.n_monosyllable_words, 90)

    def test_n_polysyllable_words(self):
        self.assertEqual(self.ts.n_polysyllable_words, 24)

    def test_flesch_kincaid_grade_level(self):
        self.assertAlmostEqual(
            self.ts.flesch_kincaid_grade_level,
            11.817647058823532,
            places=2)

    def test_flesch_readability_ease(self):
        self.assertAlmostEqual(
            self.ts.flesch_readability_ease,
            50.707745098039254,
            places=2)

    def test_smog_index(self):
        self.assertAlmostEqual(
            self.ts.smog_index,
            14.554592549557764,
            places=2)

    def test_gunning_fog_index(self):
        self.assertAlmostEqual(
            self.ts.gunning_fog_index,
            16.12549019607843,
            places=2)

    def test_coleman_liau_index(self):
        self.assertAlmostEqual(
            self.ts.coleman_liau_index,
            12.509300816176474,
            places=2)

    def test_automated_readability_index(self):
        self.assertAlmostEqual(
            self.ts.automated_readability_index,
            13.626495098039214,
            places=2)

    def test_lix(self):
        self.assertAlmostEqual(
            self.ts.lix,
            54.28431372549019,
            places=2)

    def test_gulpease_index(self):
        self.assertAlmostEqual(
            self.ts.gulpease_index,
            51.86764705882353,
            places=2)

    def test_wiener_sachtextformel(self):
        self.assertAlmostEqual(
            self.ts.wiener_sachtextformel,
            8.266410784313727,
            places=2)

    def test_basic_counts(self):
        self.assertIsInstance(self.ts.basic_counts, dict)
        basic_counts = self.ts.basic_counts
        basic_counts_keys = (
            'n_sents', 'n_words', 'n_chars', 'n_syllables', 'n_unique_words',
            'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words')
        for key in basic_counts_keys:
            self.assertEqual(basic_counts[key], getattr(self.ts, key))

    def test_readability_stats(self):
        self.assertIsInstance(self.ts.basic_counts, dict)
        readability_stats = self.ts.readability_stats
        readability_stats_keys = (
            'flesch_kincaid_grade_level', 'flesch_readability_ease', 'smog_index',
            'gunning_fog_index', 'coleman_liau_index', 'automated_readability_index',
            'lix', 'gulpease_index', 'wiener_sachtextformel')
        for key in readability_stats_keys:
            self.assertEqual(readability_stats[key], getattr(self.ts, key))

    def test_readability_stats_function(self):
        self.assertIsInstance(text_stats.readability_stats(self.doc), dict)

    def test_readability_stats_lang_function(self):
        tmp_doc = Doc('Buenos días, amigo mío!', lang='es')
        self.assertRaises(
            NotImplementedError, text_stats.readability_stats, tmp_doc)

    def test_wiener_sachtextformel_variant1(self):
        self.assertEqual(
            self.ts.wiener_sachtextformel,
            text_stats.wiener_sachtextformel(
                self.ts.n_words, self.ts.n_polysyllable_words, self.ts.n_monosyllable_words,
                self.ts.n_long_words, self.ts.n_sents, variant=1))
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(
                self.ts.n_words, self.ts.n_polysyllable_words, self.ts.n_monosyllable_words,
                self.ts.n_long_words, self.ts.n_sents, variant=1),
            8.266410784313727,
            places=2)

    def test_wiener_sachtextformel_variant2(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(
                self.ts.n_words, self.ts.n_polysyllable_words, self.ts.n_monosyllable_words,
                self.ts.n_long_words, self.ts.n_sents, variant=2),
            8.916400980392158,
            places=2)

    def test_wiener_sachtextformel_variant3(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(
                self.ts.n_words, self.ts.n_polysyllable_words, self.ts.n_monosyllable_words,
                self.ts.n_long_words, self.ts.n_sents, variant=3),
            8.432423529411766,
            places=2)

    def test_wiener_sachtextformel_variant4(self):
        self.assertAlmostEqual(
            text_stats.wiener_sachtextformel(
                self.ts.n_words, self.ts.n_polysyllable_words, self.ts.n_monosyllable_words,
                self.ts.n_long_words, self.ts.n_sents, variant=4),
            9.169619607843138,
            places=2)
