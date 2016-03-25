# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals

import unittest

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from textacy.representations.vsm import build_doc_term_matrix
from textacy.texts import TextCorpus
from textacy.tm import TopicModel

class TopicModelTestCase(unittest.TestCase):

    def setUp(self):
        texts = ["Mary had a little lamb. Its fleece was white as snow.",
                 "Everywhere that Mary went the lamb was sure to go.",
                 "It followed her to school one day, which was against the rule.",
                 "It made the children laugh and play to see a lamb at school.",
                 "And so the teacher turned it out, but still it lingered near.",
                 "It waited patiently about until Mary did appear.",
                 "Why does the lamb love Mary so? The eager children cry.",
                 "Mary loves the lamb, you know, the teacher did reply."]
        textcorpus = TextCorpus.from_texts('en', texts)
        term_lists = [doc.as_terms_list(words=True, ngrams=False, named_entities=False)
                      for doc in textcorpus]
        self.doc_term_matrix, self.id_to_word = build_doc_term_matrix(
            term_lists,
            weighting='tf', normalize=False, sublinear_tf=False, smooth_idf=True,
            min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None)

    def test_n_topics(self):
        for model in ['nmf', 'lda', 'lsa']:
            self.assertEqual(TopicModel(model, n_topics=20).n_topics, 20)

    def test_init_model(self):
        expecteds = (NMF, LatentDirichletAllocation, TruncatedSVD)
        models = ('nmf', 'lda', 'lsa')
        for model, expected in zip(models, expecteds):
            self.assertTrue(isinstance(TopicModel(model).model, expected))
