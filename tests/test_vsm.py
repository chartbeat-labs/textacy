# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from scipy.sparse import coo_matrix

from textacy import Corpus
from textacy import vsm


class VSMTestCase(unittest.TestCase):

    def setUp(self):
        texts = ["Mary had a little lamb. Its fleece was white as snow.",
                 "Everywhere that Mary went the lamb was sure to go.",
                 "It followed her to school one day, which was against the rule.",
                 "It made the children laugh and play to see a lamb at school.",
                 "And so the teacher turned it out, but still it lingered near.",
                 "It waited patiently about until Mary did appear.",
                 "Why does the lamb love Mary so? The eager children cry.",
                 "Mary loves the lamb, you know, the teacher did reply."]
        corpus = Corpus('en', texts=texts)
        term_lists = [doc.to_terms_list(ngrams=1, named_entities=False, as_strings=True)
                      for doc in corpus]
        self.vectorizer = vsm.Vectorizer(
            weighting='tf', normalize=False, sublinear_tf=False, smooth_idf=True,
            min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None)
        self.doc_term_matrix = self.vectorizer.fit_transform(term_lists)
        self.idx_lamb = [
            id_ for term, id_ in self.vectorizer.vocabulary.items() if term == 'lamb'][0]
        self.idx_child = [
            id_ for term, id_ in self.vectorizer.vocabulary.items() if term == 'child'][0]

    def test_vectorizer_feature_names(self):
        expected = [
            'mary', 'little', 'lamb', 'fleece', 'white', 'snow', 'go', 'sure',
            'follow', 'school', 'day', 'rule', 'child', 'laugh', 'play', 'teacher',
            'turn', 'linger', 'near', 'wait', 'patiently', 'appear', 'love',
            'eager', 'cry', 'know', 'reply']
        self.assertEqual(self.vectorizer.feature_names, expected)

    def test_vectorizer_bad_init_params(self):
        bad_init_params = (
            {'min_df': -1},
            {'max_df': -1},
            {'max_n_terms': -1},
            {'min_ic': -1.0},
            {'vocabulary': 'foo bar bat baz'},
            )
        for bad_init_param in bad_init_params:
            with self.assertRaises(ValueError):
                vsm.Vectorizer(**bad_init_param)

    def test_get_term_freqs(self):
        term_freqs = vsm.get_term_freqs(self.doc_term_matrix, normalized=False)
        self.assertEqual(len(term_freqs), self.doc_term_matrix.shape[1])
        self.assertEqual(term_freqs.min(), 1)
        self.assertEqual(term_freqs.max(), 5)
        self.assertEqual(term_freqs[self.idx_lamb], 5)
        self.assertEqual(term_freqs[self.idx_child], 2)

    def test_get_term_freqs_normalized(self):
        term_freqs = vsm.get_term_freqs(self.doc_term_matrix, normalized=True)
        self.assertEqual(len(term_freqs), self.doc_term_matrix.shape[1])
        self.assertAlmostEqual(term_freqs.max(), 0.18518518518518517, places=4)
        self.assertAlmostEqual(term_freqs.min(), 0.037037037037037035, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_lamb], 0.18518518518518517, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_child], 0.07407407407407407, places=4)

    def test_get_term_freqs_exception(self):
        self.assertRaises(
            ValueError, vsm.get_term_freqs, coo_matrix((1, 1)).tocsr())

    def test_get_doc_freqs(self):
        doc_freqs = vsm.get_doc_freqs(self.doc_term_matrix, normalized=False)
        self.assertEqual(len(doc_freqs), self.doc_term_matrix.shape[1])
        self.assertEqual(doc_freqs.max(), 5)
        self.assertEqual(doc_freqs.min(), 1)
        self.assertEqual(doc_freqs[self.idx_lamb], 5)
        self.assertEqual(doc_freqs[self.idx_child], 2)

    def test_get_doc_freqs_normalized(self):
        doc_freqs = vsm.get_doc_freqs(self.doc_term_matrix, normalized=True)
        self.assertEqual(len(doc_freqs), self.doc_term_matrix.shape[1])
        self.assertAlmostEqual(doc_freqs.max(), 0.625, places=3)
        self.assertAlmostEqual(doc_freqs.min(), 0.125, places=3)
        self.assertAlmostEqual(doc_freqs[self.idx_lamb], 0.625, places=3)
        self.assertAlmostEqual(doc_freqs[self.idx_child], 0.250, places=3)

    def test_get_doc_freqs_exception(self):
        self.assertRaises(
            ValueError, vsm.get_doc_freqs, coo_matrix((1, 1)).tocsr())

    def test_get_information_content(self):
        ics = vsm.get_information_content(self.doc_term_matrix)
        self.assertEqual(len(ics), self.doc_term_matrix.shape[1])
        self.assertAlmostEqual(ics.max(), 0.95443, places=4)
        self.assertAlmostEqual(ics.min(), 0.54356, places=4)
        self.assertAlmostEqual(ics[self.idx_lamb], 0.95443, places=4)
        self.assertAlmostEqual(ics[self.idx_child], 0.81127, places=4)

    def test_filter_terms_by_df_identity(self):
        dtm, vocab = vsm.filter_terms_by_df(self.doc_term_matrix, self.vectorizer.vocabulary,
                                            max_df=1.0, min_df=1, max_n_terms=None)
        self.assertEqual(dtm.shape, self.doc_term_matrix.shape)
        self.assertEqual(vocab, self.vectorizer.vocabulary)

    def test_filter_terms_by_df_max_n_terms(self):
        dtm, vocab = vsm.filter_terms_by_df(self.doc_term_matrix, self.vectorizer.vocabulary,
                                            max_df=1.0, min_df=1, max_n_terms=2)
        self.assertEqual(dtm.shape, (8, 2))
        self.assertEqual(sorted(vocab.keys()), ['lamb', 'mary'])

    def test_filter_terms_by_df_min_df(self):
        dtm, vocab = vsm.filter_terms_by_df(self.doc_term_matrix, self.vectorizer.vocabulary,
                                            max_df=1.0, min_df=2, max_n_terms=None)
        self.assertEqual(dtm.shape, (8, 6))
        self.assertEqual(
            sorted(vocab.keys()),
            ['child', 'lamb', 'love', 'mary', 'school', 'teacher'])

    def test_filter_terms_by_df_exception(self):
        self.assertRaises(ValueError, vsm.filter_terms_by_df,
                          self.doc_term_matrix, self.vectorizer.vocabulary,
                          max_df=1.0, min_df=6, max_n_terms=None)

    def test_filter_terms_by_ic_identity(self):
        dtm, vocab = vsm.filter_terms_by_ic(self.doc_term_matrix, self.vectorizer.vocabulary,
                                            min_ic=0.0, max_n_terms=None)
        self.assertEqual(dtm.shape, self.doc_term_matrix.shape)
        self.assertEqual(vocab, self.vectorizer.vocabulary)

    def test_filter_terms_by_ic_max_n_terms(self):
        dtm, vocab = vsm.filter_terms_by_ic(self.doc_term_matrix, self.vectorizer.vocabulary,
                                            min_ic=0.0, max_n_terms=3)
        self.assertEqual(dtm.shape, (8, 3))
        self.assertEqual(len(vocab), 3)
