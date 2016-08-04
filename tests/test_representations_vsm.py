# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

import numpy as np
from scipy.sparse import coo_matrix

from textacy import Corpus
from textacy.representations import vsm


class RepresentationsVSMTestCase(unittest.TestCase):

    def setUp(self):
        texts = ["Burton loves to work with data — especially text data.",
                 "Extracting information from unstructured text is an interesting challenge.",
                 "Sometimes the hardest part is acquiring the right text; as with much analysis, it's garbage in, garbage out."]
        textcorpus = Corpus.from_texts('en', texts)
        term_lists = [doc.as_terms_list(words=True, ngrams=False, named_entities=False)
                      for doc in textcorpus]
        self.doc_term_matrix, self.id_to_word = vsm.build_doc_term_matrix(
            term_lists,
            weighting='tf', normalize=False, sublinear_tf=False, smooth_idf=True,
            min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None)
        self.idx_text = [k for k, v in self.id_to_word.items() if v == 'text'][0]
        self.idx_garbage = [k for k, v in self.id_to_word.items() if v == 'garbage'][0]

    def test_get_term_freqs(self):
        term_freqs = vsm.get_term_freqs(self.doc_term_matrix, normalized=False)
        self.assertEqual(len(term_freqs), self.doc_term_matrix.shape[1])
        self.assertEqual(term_freqs.max(), 3.0)
        self.assertEqual(term_freqs.min(), 1.0)
        self.assertEqual(term_freqs[self.idx_text], 3.0)
        self.assertEqual(term_freqs[self.idx_garbage], 2.0)

    def test_get_term_freqs_normalized(self):
        term_freqs = vsm.get_term_freqs(self.doc_term_matrix, normalized=True)
        self.assertEqual(len(term_freqs), self.doc_term_matrix.shape[1])
        self.assertAlmostEqual(term_freqs.max(), 0.1765, places=4)
        self.assertAlmostEqual(term_freqs.min(), 0.05882, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_text], 0.1765, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_garbage], 0.1176, places=4)

    def test_get_term_freqs_exception(self):
        self.assertRaises(
            ValueError, vsm.get_term_freqs, coo_matrix((1,1)).tocsr())

    def test_get_doc_freqs(self):
        doc_freqs = vsm.get_doc_freqs(self.doc_term_matrix, normalized=False)
        self.assertEqual(len(doc_freqs), self.doc_term_matrix.shape[1])
        self.assertEqual(doc_freqs.max(), 3)
        self.assertEqual(doc_freqs.min(), 1)
        self.assertEqual(doc_freqs[self.idx_text], 3)
        self.assertEqual(doc_freqs[self.idx_garbage], 1)

    def test_get_doc_freqs_normalized(self):
        doc_freqs = vsm.get_doc_freqs(self.doc_term_matrix, normalized=True)
        self.assertEqual(len(doc_freqs), self.doc_term_matrix.shape[1])
        self.assertEqual(doc_freqs.max(), 1.0)
        self.assertAlmostEqual(doc_freqs.min(), 0.3333, places=4)
        self.assertEqual(doc_freqs[self.idx_text], 1.0)
        self.assertAlmostEqual(doc_freqs[self.idx_garbage], 0.3333, places=4)

    def test_get_doc_freqs_exception(self):
        self.assertRaises(
            ValueError, vsm.get_doc_freqs, coo_matrix((1,1)).tocsr())

    def test_get_information_content(self):
        ics = vsm.get_information_content(self.doc_term_matrix)
        self.assertEqual(len(ics), self.doc_term_matrix.shape[1])
        self.assertAlmostEqual(ics.max(), 0.9183, places=4)
        self.assertEqual(ics.min(), 0.0)
        self.assertEqual(ics[self.idx_text], 0.0)
        self.assertAlmostEqual(ics[self.idx_garbage], 0.9183, places=4)

    def test_filter_terms_by_df_identity(self):
        dtm, i2w = vsm.filter_terms_by_df(self.doc_term_matrix, self.id_to_word,
                                          max_df=1.0, min_df=1, max_n_terms=None)
        self.assertEqual(dtm.shape, self.doc_term_matrix.shape)
        self.assertEqual(i2w, self.id_to_word)

    def test_filter_terms_by_df_max_n_terms(self):
        dtm, i2w = vsm.filter_terms_by_df(self.doc_term_matrix, self.id_to_word,
                                          max_df=1.0, min_df=1, max_n_terms=1)
        self.assertEqual(dtm.shape, (3, 1))
        self.assertEqual(i2w, {0: 'text'})

    def test_filter_terms_by_df_min_df(self):
        dtm, i2w = vsm.filter_terms_by_df(self.doc_term_matrix, self.id_to_word,
                                                 max_df=1.0, min_df=2, max_n_terms=None)
        self.assertEqual(dtm.shape, (3, 1))
        self.assertEqual(sorted(i2w.values()), ['text'])

    def test_filter_terms_by_df_exception(self):
        self.assertRaises(ValueError, vsm.filter_terms_by_df,
                          self.doc_term_matrix, self.id_to_word,
                          max_df=0.25, min_df=1, max_n_terms=None)

    def test_filter_terms_by_ic_identity(self):
        dtm, i2w = vsm.filter_terms_by_ic(self.doc_term_matrix, self.id_to_word,
                                                 min_ic=0.0, max_n_terms=None)
        self.assertEqual(dtm.shape, self.doc_term_matrix.shape)
        self.assertEqual(i2w, self.id_to_word)

    def test_filter_terms_by_ic_max_n_terms(self):
        dtm, i2w = vsm.filter_terms_by_ic(self.doc_term_matrix, self.id_to_word,
                                                 min_ic=0.0, max_n_terms=1)
        self.assertEqual(dtm.shape, (3, 1))
        self.assertEqual(len(i2w), 1)
