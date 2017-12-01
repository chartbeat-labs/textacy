from __future__ import absolute_import, unicode_literals

from operator import itemgetter
import unittest

import numpy as np
import scipy.sparse as sp
from spacy.tokens.span import Span as SpacySpan

import textacy
import textacy.datasets


class ReadmeTestCase(unittest.TestCase):

    def setUp(self):
        self.spacy_lang = textacy.data.load_spacy('en')
        self.cw = textacy.datasets.CapitolWords()
        self.text = list(self.cw.texts(speaker_name={'Bernie Sanders'}, limit=1))[0]
        self.doc = textacy.Doc(self.text.strip(), lang=self.spacy_lang)
        records = self.cw.records(speaker_name={'Bernie Sanders'}, limit=10)
        text_stream, metadata_stream = textacy.fileio.split_record_fields(
            records, 'text')
        self.corpus = textacy.Corpus(
            self.spacy_lang, texts=text_stream, metadatas=metadata_stream)

    def test_streaming_functionality(self):
        self.assertIsInstance(self.cw, textacy.datasets.base.Dataset)
        self.assertIsInstance(self.corpus, textacy.Corpus)

    def test_vectorization_and_topic_modeling_functionality(self):
        n_topics = 10
        top_n = 10
        vectorizer = textacy.Vectorizer(
            weighting='tfidf', normalize=True, smooth_idf=True,
            min_df=2, max_df=0.95)
        doc_term_matrix = vectorizer.fit_transform(
            (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
             for doc in self.corpus))
        model = textacy.TopicModel('nmf', n_topics=n_topics)
        model.fit(doc_term_matrix)
        doc_topic_matrix = model.transform(doc_term_matrix)
        self.assertIsInstance(doc_term_matrix, sp.csr_matrix)
        self.assertIsInstance(doc_topic_matrix, np.ndarray)
        self.assertEqual(doc_topic_matrix.shape[1], n_topics)
        for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=top_n):
            self.assertIsInstance(topic_idx, int)
            self.assertEqual(len(top_terms), top_n)

    def test_corpus_functionality(self):
        self.assertIsInstance(self.corpus[0], textacy.Doc)
        self.assertTrue(
            list(self.corpus.get(
                lambda doc: doc.metadata['speaker_name'] == 'Bernie Sanders'))
            )

    def test_plaintext_functionality(self):
        preprocessed_text = textacy.preprocess_text(
            self.text, lowercase=True, no_punct=True)[:100]
        self.assertTrue(
            all(char.islower() for char in preprocessed_text if char.isalpha()))
        self.assertTrue(
            all(char.isalnum() or char.isspace() for char in preprocessed_text))
        keyword = 'America'
        kwics = textacy.text_utils.keyword_in_context(
            self.text, keyword, window_width=35, print_only=False)
        for pre, kw, post in kwics:
            self.assertEqual(kw, keyword)
            self.assertIsInstance(pre, textacy.compat.unicode_)
            self.assertIsInstance(post, textacy.compat.unicode_)

    def test_extract_functionality(self):
        bigrams = list(textacy.extract.ngrams(
            self.doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))[:10]
        for bigram in bigrams:
            self.assertIsInstance(bigram, SpacySpan)
            self.assertEqual(len(bigram), 2)

        trigrams = list(textacy.extract.ngrams(
            self.doc, 3, filter_stops=True, filter_punct=True, min_freq=2))[:10]
        for trigram in trigrams:
            self.assertIsInstance(trigram, SpacySpan)
            self.assertEqual(len(trigram), 3)

        nes = list(textacy.extract.named_entities(
            self.doc, drop_determiners=False, exclude_types='numeric'))[:10]
        for ne in nes:
            self.assertIsInstance(ne, SpacySpan)
            self.assertTrue(ne.label_)
            self.assertNotEqual(ne.label_, 'QUANTITY')

        pos_regex_matches = list(textacy.extract.pos_regex_matches(
            self.doc, textacy.constants.POS_REGEX_PATTERNS['en']['NP']))[:10]
        for match in pos_regex_matches:
            self.assertIsInstance(match, SpacySpan)

        stmts = list(textacy.extract.semistructured_statements(
            self.doc, 'I', cue='be'))[:10]
        for stmt in stmts:
            self.assertIsInstance(stmt, list)
            self.assertIsInstance(stmt[0], textacy.compat.unicode_)
            self.assertEqual(len(stmt), 3)

        keyterms = textacy.keyterms.textrank(
            self.doc, n_keyterms=10)
        for keyterm in keyterms:
            self.assertIsInstance(keyterm, tuple)
            self.assertIsInstance(keyterm[0], textacy.compat.unicode_)
            self.assertIsInstance(keyterm[1], float)
            self.assertTrue(keyterm[1] > 0.0)

    def test_text_stats_functionality(self):
        ts = textacy.TextStats(self.doc)

        self.assertIsInstance(ts.n_words, int)
        self.assertIsInstance(ts.flesch_kincaid_grade_level, float)

        basic_counts = ts.basic_counts
        self.assertIsInstance(basic_counts, dict)
        for field in ('n_chars', 'n_words', 'n_sents'):
            self.assertIsInstance(basic_counts.get(field), int)

        readability_stats = ts.readability_stats
        self.assertIsInstance(readability_stats, dict)
        for field in ('flesch_kincaid_grade_level', 'automated_readability_index', 'wiener_sachtextformel'):
            self.assertIsInstance(readability_stats.get(field), float)
