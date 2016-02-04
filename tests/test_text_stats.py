from __future__ import absolute_import, unicode_literals

import numpy as np
from scipy.sparse import coo_matrix
import unittest

from textacy import text_stats
from textacy.texts import TextCorpus


class TextStatsTestCase(unittest.TestCase):

    def setUp(self):
        texts = ["Burton loves to work with data — especially text data.",
                 "Extracting information from unstructured text is an interesting challenge.",
                 "Sometimes the hardest part is acquiring the right text; as with much data analysis, it's garbage in, garbage out."]
        self.textcorpus = TextCorpus.from_texts(texts, lang='en')
        self.term_doc_matrix, self.id_to_word = self.textcorpus.to_term_doc_matrix(
            weighting='tf', normalize=False, binarize=False, smooth_idf=True,
            min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None,
            ngram_range=(1, 1), include_nes=False,
            include_nps=False, include_kts=False)
        self.idx_text = [k for k, v in self.id_to_word.items() if v == 'text'][0]
        self.idx_garbage = [k for k, v in self.id_to_word.items() if v == 'garbage'][0]
        # for testing the readability statistics
        self.n_chars = 2855
        self.n_syllables = 857
        self.n_words = 441
        self.n_polysyllable_words = 104
        self.n_sents = 21

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

    def test_get_term_freqs(self):
        term_freqs = text_stats.get_term_freqs(self.term_doc_matrix, normalized=False)
        self.assertEqual(len(term_freqs), self.term_doc_matrix.shape[1])
        self.assertEqual(term_freqs.max(), 3.0)
        self.assertEqual(term_freqs.min(), 1.0)
        self.assertEqual(term_freqs[self.idx_text], 3.0)
        self.assertEqual(term_freqs[self.idx_garbage], 2.0)

    def test_get_term_freqs_normalized(self):
        term_freqs = text_stats.get_term_freqs(self.term_doc_matrix, normalized=True)
        self.assertEqual(len(term_freqs), self.term_doc_matrix.shape[1])
        self.assertAlmostEqual(term_freqs.max(), 0.1765, places=4)
        self.assertAlmostEqual(term_freqs.min(), 0.0588, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_text], 0.1765, places=4)
        self.assertAlmostEqual(term_freqs[self.idx_garbage], 0.1176, places=4)

    def test_get_term_freqs_exception(self):
        self.assertRaises(
            ValueError, text_stats.get_term_freqs, coo_matrix((1,1)).tocsr())

    def test_get_doc_freqs(self):
        doc_freqs = text_stats.get_doc_freqs(self.term_doc_matrix, normalized=False)
        self.assertEqual(len(doc_freqs), self.term_doc_matrix.shape[1])
        self.assertEqual(doc_freqs.max(), 3)
        self.assertEqual(doc_freqs.min(), 1)
        self.assertEqual(doc_freqs[self.idx_text], 3)
        self.assertEqual(doc_freqs[self.idx_garbage], 1)

    def test_get_doc_freqs_normalized(self):
        doc_freqs = text_stats.get_doc_freqs(self.term_doc_matrix, normalized=True)
        self.assertEqual(len(doc_freqs), self.term_doc_matrix.shape[1])
        self.assertEqual(doc_freqs.max(), 1.0)
        self.assertAlmostEqual(doc_freqs.min(), 0.3333, places=4)
        self.assertEqual(doc_freqs[self.idx_text], 1.0)
        self.assertAlmostEqual(doc_freqs[self.idx_garbage], 0.3333, places=4)

    def test_get_doc_freqs_exception(self):
        self.assertRaises(
            ValueError, text_stats.get_doc_freqs, coo_matrix((1,1)).tocsr())

    def test_get_information_content(self):
        ics = text_stats.get_information_content(self.term_doc_matrix)
        self.assertEqual(len(ics), self.term_doc_matrix.shape[1])
        self.assertAlmostEqual(ics.max(), 0.9183, places=4)
        self.assertEqual(ics.min(), 0.0)
        self.assertEqual(ics[self.idx_text], 0.0)
        self.assertAlmostEqual(ics[self.idx_garbage], 0.9183, places=4)

    def test_filter_terms_by_df_identity(self):
        tdm, i2w = text_stats.filter_terms_by_df(self.term_doc_matrix, self.id_to_word,
                                                 max_df=1.0, min_df=1, max_n_terms=None)
        self.assertEqual(tdm.shape, self.term_doc_matrix.shape)
        self.assertEqual(i2w, self.id_to_word)

    def test_filter_terms_by_df_max_n_terms(self):
        tdm, i2w = text_stats.filter_terms_by_df(self.term_doc_matrix, self.id_to_word,
                                                 max_df=1.0, min_df=1, max_n_terms=1)
        self.assertEqual(tdm.shape, (3, 1))
        self.assertEqual(i2w, {0: 'text'})

    def test_filter_terms_by_df_min_df(self):
        tdm, i2w = text_stats.filter_terms_by_df(self.term_doc_matrix, self.id_to_word,
                                                 max_df=1.0, min_df=2, max_n_terms=None)
        self.assertEqual(tdm.shape, (3, 2))
        self.assertEqual(i2w, {0: 'text', 1: 'data'})

    def test_filter_terms_by_df_exception(self):
        self.assertRaises(ValueError, text_stats.filter_terms_by_df,
                          self.term_doc_matrix, self.id_to_word,
                          max_df=0.25, min_df=1, max_n_terms=None)

    def test_filter_terms_by_ic_identity(self):
        tdm, i2w = text_stats.filter_terms_by_ic(self.term_doc_matrix, self.id_to_word,
                                                 min_ic=0.0, max_n_terms=None)
        self.assertEqual(tdm.shape, self.term_doc_matrix.shape)
        self.assertEqual(i2w, self.id_to_word)

    def test_filter_terms_by_ic_max_n_terms(self):
        tdm, i2w = text_stats.filter_terms_by_ic(self.term_doc_matrix, self.id_to_word,
                                                 min_ic=0.0, max_n_terms=1)
        self.assertEqual(tdm.shape, (3, 1))
        self.assertEqual(i2w, {0: 'garbage'})
