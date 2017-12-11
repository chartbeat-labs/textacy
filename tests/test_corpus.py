# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import Corpus
from textacy import Doc
from textacy import cache
from textacy import compat
from textacy import fileio
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filename is None,
    reason='CapitolWords dataset must be downloaded before running tests')


class TestCorpusInit(object):

    def test_corpus_init_lang(self):
        assert isinstance(Corpus('en'), Corpus)
        assert isinstance(Corpus(cache.load_spacy('en')), Corpus)
        for bad_lang in (b'en', None):
            with pytest.raises(TypeError):
                Corpus(bad_lang)

    def test_corpus_init_texts(self):
        limit = 3
        corpus = Corpus('en', texts=DATASET.texts(limit=limit))
        assert len(corpus.docs) == limit
        assert all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus)

    def test_corpus_init_texts_and_metadatas(self):
        limit = 3
        texts, metadatas = fileio.split_record_fields(
            DATASET.records(limit=limit), 'text')
        texts = list(texts)
        metadatas = list(metadatas)
        corpus = Corpus('en', texts=texts, metadatas=metadatas)
        assert len(corpus.docs) == limit
        assert all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus)
        for i in range(limit):
            assert texts[i] == corpus[i].text
            assert metadatas[i] == corpus[i].metadata

    def test_corpus_init_docs(self):
        limit = 3
        texts, metadatas = fileio.split_record_fields(
            DATASET.records(limit=limit), 'text')
        docs = [Doc(text, lang='en', metadata=metadata)
                for text, metadata in zip(texts, metadatas)]
        corpus = Corpus('en', docs=docs)
        assert len(corpus.docs) == limit
        assert all(doc.spacy_vocab is corpus.spacy_vocab for doc in corpus)
        for i in range(limit):
            assert corpus[i].metadata == docs[i].metadata
        corpus = Corpus(
            'en', docs=docs, metadatas=({'foo': 'bar'} for _ in range(limit)))
        for i in range(limit):
            assert corpus[i].metadata == {'foo': 'bar'}


@pytest.fixture(scope='module')
def corpus(request):
    texts, metadatas = fileio.split_record_fields(
        DATASET.records(limit=3), 'text')
    corpus = Corpus('en', texts=texts, metadatas=metadatas)
    return corpus


def test_corpus_save_and_load(tmpdir, corpus):
    filepath = str(tmpdir.join('test_corpus_save_and_load.pkl'))
    corpus.save(filepath)
    new_corpus = Corpus.load(filepath)
    assert isinstance(new_corpus, Corpus)
    assert len(new_corpus) == len(corpus)
    assert new_corpus.lang == corpus.lang
    assert new_corpus.spacy_lang.pipe_names == corpus.spacy_lang.pipe_names
    assert new_corpus[0].spacy_doc.user_data['textacy'].get('spacy_lang_meta') is None
    for i in range(len(new_corpus)):
        assert new_corpus[i].metadata == corpus[i].metadata
