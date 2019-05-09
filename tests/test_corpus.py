# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

import spacy

from textacy import Corpus
from textacy import Doc
from textacy import cache
from textacy import compat
from textacy import io
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def corpus(request):
    return Corpus("en", data=DATASET.records(limit=3))


class TestCorpusInit(object):

    def test_corpus_init_lang(self):
        assert isinstance(Corpus("en"), Corpus)
        assert isinstance(Corpus(cache.load_spacy("en")), Corpus)
        for bad_lang in (b"en", None):
            with pytest.raises(TypeError):
                Corpus(bad_lang)

    def test_corpus_init_texts(self):
        limit = 3
        texts = list(DATASET.texts(limit=limit))
        corpus = Corpus("en", data=texts)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(text == doc.text for text, doc in zip(texts, corpus))

    def test_corpus_init_records(self):
        limit = 3
        records = list(DATASET.records(limit=limit))
        corpus = Corpus("en", data=records)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(record[0] == doc.text for record, doc in zip(records, corpus))
        # TODO: handle document metadata!
        # assert all(record[1] == doc._.meta for record, doc in zip(records, corpus))

    def test_corpus_init_docs(self):
        limit = 3
        spacy_lang = cache.load_spacy("en")
        texts = DATASET.texts(limit=limit)
        docs = [spacy_lang(text) for text in texts]
        corpus = Corpus("en", data=docs)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(doc1 is doc2 for doc1, doc2 in zip(docs, corpus))

    def test_corpus_init_no_parser(self):
        spacy_lang = cache.load_spacy("en", disable=("parser",))
        corpus = Corpus(spacy_lang, data=(spacy_lang("This is a sentence in a doc."),))
        assert len(corpus) == 1
        assert corpus.n_sents == 0


def test_corpus_save_and_load(tmpdir, corpus):
    filepath = str(tmpdir.join("test_corpus_save_and_load.pkl"))
    corpus.save(filepath)
    loaded_corpus = Corpus.load("en", filepath)
    assert isinstance(loaded_corpus, Corpus)
    assert len(loaded_corpus) == len(corpus)
    assert loaded_corpus.spacy_lang.meta == corpus.spacy_lang.meta
    assert loaded_corpus.spacy_lang.pipe_names == corpus.spacy_lang.pipe_names


# TODO: add more tests :)
