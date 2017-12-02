# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import Corpus
from textacy import Doc
from textacy import compat
from textacy import data
from textacy import fileio
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()


@unittest.skipUnless(
    DATASET.filename, 'CapitolWords dataset must be downloaded before running tests')
class CorpusInitTestCase(unittest.TestCase):

    def test_corpus_init_lang(self):
        self.assertIsInstance(Corpus('en'), Corpus)
        self.assertIsInstance(Corpus(data.load_spacy('en')), Corpus)
        for bad_lang in (b'en', None):
            with self.assertRaises(TypeError):
                Corpus(bad_lang)

    def test_corpus_init_texts(self):
        limit = 3
        corpus = Corpus('en', texts=DATASET.texts(limit=limit))
        self.assertEqual(len(corpus.docs), limit)
        self.assertTrue(
            all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus))

    def test_corpus_init_texts_and_metadatas(self):
        limit = 3
        texts, metadatas = fileio.split_record_fields(
            DATASET.records(limit=limit), 'text')
        texts = list(texts)
        metadatas = list(metadatas)
        corpus = Corpus('en', texts=texts, metadatas=metadatas)
        self.assertEqual(len(corpus.docs), limit)
        self.assertTrue(
            all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus))
        for i in range(limit):
            self.assertEqual(texts[i], corpus[i].text)
            self.assertEqual(metadatas[i], corpus[i].metadata)

    def test_corpus_init_docs(self):
        limit = 3
        texts, metadatas = fileio.split_record_fields(
            DATASET.records(limit=limit), 'text')
        docs = [Doc(text, lang='en', metadata=metadata)
                for text, metadata in zip(texts, metadatas)]
        corpus = Corpus('en', docs=docs)
        self.assertEqual(len(corpus.docs), limit)
        self.assertTrue(
            all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus))
        for i in range(limit):
            self.assertEqual(corpus[i].metadata, docs[i].metadata)
        corpus = Corpus(
            'en', docs=docs, metadatas=({'foo': 'bar'} for _ in range(limit)))
        for i in range(limit):
            self.assertEqual(corpus[i].metadata, {'foo': 'bar'})


class CorpusMethodsTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpus', dir=os.path.dirname(os.path.abspath(__file__)))
        texts, metadatas = fileio.split_record_fields(
            DATASET.records(limit=3), 'text')
        self.corpus = Corpus('en', texts=texts, metadatas=metadatas)

    def test_corpus_save_and_load(self):
        filepath = os.path.join(self.tempdir, 'test_corpus_save_and_load.pkl')
        self.corpus.save(filepath)
        new_corpus = Corpus.load(filepath)
        self.assertIsInstance(new_corpus, Corpus)
        self.assertEqual(len(new_corpus), len(self.corpus))
        self.assertEqual(new_corpus.lang, self.corpus.lang)
        self.assertEqual(
            new_corpus.spacy_lang.pipe_names,
            self.corpus.spacy_lang.pipe_names)
        self.assertIsNone(
            new_corpus[0].spacy_doc.user_data['textacy'].get('spacy_lang_meta'))
        for i in range(len(new_corpus)):
            self.assertEqual(new_corpus[i].metadata, self.corpus[i].metadata)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
