from __future__ import absolute_import, unicode_literals

import unittest

import textacy


class DistanceTestCase(unittest.TestCase):

    def setUp(self):
        self.text1 = 'She spoke to the assembled journalists.'
        self.text2 = 'He chatted with the gathered press.'
        self.doc1 = textacy.Document(self.text1, lang='en')
        self.doc2 = textacy.Document(self.text2, lang='en')

    def test_word_movers(self):
        metrics = ('cosine', 'l1', 'manhattan', 'l2', 'euclidean')
        expected_values = (0.467695, 0.655712, 0.655712, 0.668999, 0.668999)
        for metric, expected_value in zip(metrics, expected_values):
            self.assertAlmostEqual(
                textacy.distance.word_movers(self.doc1, self.doc2, metric=metric),
                expected_value,
                places=4)

    def test_word2vec(self):
        pairs = ((self.doc1, self.doc2),
                 (self.doc1[-2:], self.doc2[-2:]),
                 (self.doc1[-1], self.doc2[-1]))
        expected_values = (0.089036, 0.238299, 0.500000)
        for pair, expected_value in zip(pairs, expected_values):
            self.assertAlmostEqual(
                textacy.distance.word2vec(pair[0], pair[1]),
                expected_value,
                places=4)

    def test_jaccard(self):
        pairs = ((self.text1, self.text2),
                 (self.text1.split(), self.text2.split()))
        expected_values = (0.541666, 0.909090)
        for pair, expected_value in zip(pairs, expected_values):
            self.assertAlmostEqual(
                textacy.distance.jaccard(pair[0], pair[1]),
                expected_value,
                places=4)

    def test_jaccard_exception(self):
        self.assertRaises(
            ValueError, textacy.distance.jaccard,
            self.text1, self.text2, True)

    def test_jaccard_fuzzy_match(self):
        thresholds = (50, 70, 90)
        expected_values = (0.545454, 0.727272, 0.909090)
        for thresh, expected_value in zip(thresholds, expected_values):
            self.assertAlmostEqual(
                textacy.distance.jaccard(self.text1.split(), self.text2.split(),
                                         fuzzy_match=True, match_threshold=thresh),
                expected_value,
                places=4)

    def test_hamming(self):
        self.assertEqual(
            textacy.distance.hamming(self.text1, self.text2),
            34)
        self.assertEqual(
            textacy.distance.hamming(self.text1, self.text2, normalize=True),
            0.8717948717948718)

    def test_levenshtein(self):
        self.assertEqual(
            textacy.distance.levenshtein(self.text1, self.text2),
            25)
        self.assertEqual(
            textacy.distance.levenshtein(self.text1, self.text2, normalize=True),
            0.6410256410256411)

    def test_jaro_winkler(self):
        self.assertEqual(
            textacy.distance.jaro_winkler(self.text1, self.text2),
            0.4281995781995781)
