from __future__ import absolute_import, unicode_literals

import unittest

import textacy
from textacy.compat import is_python2


class SimilarityTestCase(unittest.TestCase):

    def setUp(self):
        self.text1 = 'She spoke to the assembled journalists.'
        self.text2 = 'He chatted with the gathered press.'
        self.doc1 = textacy.Doc(self.text1, lang='en')
        self.doc2 = textacy.Doc(self.text2, lang='en')

    def test_word_movers(self):
        metrics = ('cosine', 'l1', 'manhattan', 'l2', 'euclidean')
        expected_values = (0.27088, 0.112838, 0.112838, 0.102415, 0.102415)
        for metric, expected_value in zip(metrics, expected_values):
            self.assertAlmostEqual(
                textacy.similarity.word_movers(self.doc1, self.doc2, metric=metric),
                expected_value,
                places=4)

    def test_word2vec(self):
        pairs = ((self.doc1, self.doc2),
                 (self.doc1[-2:], self.doc2[-2:]),
                 (self.doc1[-1], self.doc2[-1]))
        expected_values = (0.893582, 0.712395, 1.000000)
        for pair, expected_value in zip(pairs, expected_values):
            self.assertAlmostEqual(
                textacy.similarity.word2vec(pair[0], pair[1]),
                expected_value,
                places=4)

    def test_jaccard(self):
        pairs = ((self.text1, self.text2),
                 (self.text1.split(), self.text2.split()))
        expected_values = (0.4583333, 0.09091)
        for pair, expected_value in zip(pairs, expected_values):
            self.assertAlmostEqual(
                textacy.similarity.jaccard(pair[0], pair[1]),
                expected_value,
                places=4)

    def test_jaccard_exception(self):
        self.assertRaises(
            ValueError, textacy.similarity.jaccard,
            self.text1, self.text2, True)

    def test_jaccard_fuzzy_match(self):
        thresholds = (0.50, 0.70, 0.90)
        expected_values = (0.454546, 0.272728, 0.09091)
        for thresh, expected_value in zip(thresholds, expected_values):
            self.assertAlmostEqual(
                textacy.similarity.jaccard(self.text1.split(), self.text2.split(),
                                           fuzzy_match=True, match_threshold=thresh),
                expected_value,
                places=4)

    @unittest.skipIf(is_python2, "Python 2's unittest doesn't have ``assertWarns``")
    def test_jaccard_fuzzy_match_warning(self):
        thresh = 50
        with self.assertWarns(UserWarning):
            textacy.similarity.jaccard(
                self.text1.split(), self.text2.split(),
                fuzzy_match=True, match_threshold=thresh)

    def test_hamming(self):
        self.assertEqual(
            textacy.similarity.hamming(self.text1, self.text2),
            0.1282051282051282)

    def test_levenshtein(self):
        self.assertEqual(
            textacy.similarity.levenshtein(self.text1, self.text2),
            0.3589743589743589)

    def test_jaro_winkler(self):
        self.assertEqual(
            textacy.similarity.jaro_winkler(self.text1, self.text2),
            0.5718004218004219)
