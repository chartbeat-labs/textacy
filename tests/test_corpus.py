# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import numpy as np
import pytest
import spacy

from textacy import Corpus
from textacy import cache
from textacy import compat
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def corpus():
    return Corpus("en", data=DATASET.records(limit=5))


class TestCorpusInit(object):

    def test_corpus_init_lang(self):
        assert isinstance(Corpus("en"), Corpus)
        assert isinstance(Corpus(cache.load_spacy_lang("en")), Corpus)
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
        assert all(record[1] == doc._.meta for record, doc in zip(records, corpus))

    def test_corpus_init_docs(self):
        limit = 3
        spacy_lang = cache.load_spacy_lang("en")
        texts = DATASET.texts(limit=limit)
        docs = [spacy_lang(text) for text in texts]
        corpus = Corpus("en", data=docs)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(doc1 is doc2 for doc1, doc2 in zip(docs, corpus))

    def test_corpus_init_no_parser(self):
        spacy_lang = cache.load_spacy_lang("en", disable=("parser",))
        corpus = Corpus(spacy_lang, data=(spacy_lang("This is a sentence in a doc."),))
        assert len(corpus) == 1
        assert corpus.n_sents == 0


class TestCorpusDunder(object):

    def test_repr(self, corpus):
        repr = compat.unicode_(corpus)
        assert repr.startswith("Corpus")
        assert all("{}".format(n) in repr for n in [corpus.n_docs, corpus.n_tokens])

    def test_len(self, corpus):
        assert isinstance(len(corpus), int)
        assert len(corpus) == len(corpus.docs) == corpus.n_docs

    def test_iter(self, corpus):
        assert isinstance(corpus, compat.Iterable)

    def test_getitem(self, corpus):
        assert isinstance(corpus[0], spacy.tokens.Doc)
        assert all(isinstance(doc, spacy.tokens.Doc) for doc in corpus[0:2])

    def test_delitem(self, corpus):
        del corpus[-1]
        del corpus[-1:]
        with pytest.raises(TypeError):
            del corpus["foo"]


class TestCorpusProperties(object):

    def test_vectors(self, corpus):
        vectors = corpus.vectors
        assert isinstance(vectors, np.ndarray)
        assert len(vectors.shape) == 2
        assert vectors.shape[0] == len(corpus)

    def test_vector_norms(self, corpus):
        vector_norms = corpus.vector_norms
        assert isinstance(vector_norms, np.ndarray)
        assert len(vector_norms.shape) == 2
        assert vector_norms.shape[0] == len(corpus)


class TestCorpusMethods(object):

    def test_corpus_add(self, corpus):
        spacy_lang = cache.load_spacy_lang("en")
        datas = (
            "This is an english sentence.",
            ("This is an english sentence.", {"foo": "bar"}),
            spacy_lang("This is an english sentence."),
            ["This is one sentence.", "This is another sentence."],
            [("This is sentence #1.", {"foo": "bar"}), ("This is sentence #2.", {"bat": "baz"})],
            [spacy_lang("This is sentence #1"), spacy_lang("This is sentence #2")],
        )
        n_docs = corpus.n_docs
        for data in datas:
            corpus.add(data)
            assert corpus.n_docs > n_docs
            n_docs = corpus.n_docs

    def test_corpus_add_typeerror(self, corpus):
        datas = (
            b"This is a byte string.",
            [b"This is a byte string.", b"This is another byte string."],
        )
        for data in datas:
            with pytest.raises(TypeError):
                corpus.add(data)

    def test_corpus_get(self, corpus):
        match_funcs = (
            lambda doc: True,
            lambda doc: doc._.meta.get("speaker_name") == "Bernie Sanders",
        )
        for match_func in match_funcs:
            assert len(list(corpus.get(match_func))) > 0
            assert len(list(corpus.get(match_func, limit=1))) == 1

    def test_corpus_remove(self, corpus):
        match_funcs = (
            lambda doc: doc._.meta.get("foo") == "bar",
            lambda doc: len(doc) < 10,
        )
        n_docs = corpus.n_docs
        for match_func in match_funcs[:1]:
            corpus.remove(match_func)
            assert corpus.n_docs < n_docs
            assert not any([match_func(doc) for doc in corpus])
            n_docs = corpus.n_docs

    def test_corpus_word_counts(self, corpus):
        abs_counts = corpus.word_counts(weighting="count", normalize="lower")
        rel_counts = corpus.word_counts(weighting="freq", normalize="lower")
        assert isinstance(abs_counts, dict)
        assert all(isinstance(count, int) for count in abs_counts.values())
        assert min(abs_counts.values()) > 0
        assert isinstance(rel_counts, dict)
        assert all(isinstance(count, float) for count in rel_counts.values())
        assert min(rel_counts.values()) > 0 and max(rel_counts.values()) <= 1

    def test_corpus_word_counts_error(self, corpus):
        with pytest.raises(ValueError):
            corpus.word_counts(weighting="foo")

    def test_corpus_word_doc_counts(self, corpus):
        abs_counts = corpus.word_doc_counts(weighting="count", normalize="lower")
        rel_counts = corpus.word_doc_counts(weighting="freq", normalize="lower")
        inv_counts = corpus.word_doc_counts(weighting="idf", normalize="lower")
        assert isinstance(abs_counts, dict)
        assert all(isinstance(count, int) for count in abs_counts.values())
        assert min(abs_counts.values()) > 0
        assert isinstance(rel_counts, dict)
        assert all(isinstance(count, float) for count in rel_counts.values())
        assert min(rel_counts.values()) > 0 and max(rel_counts.values()) <= 1
        assert isinstance(inv_counts, dict)
        assert min(inv_counts.values()) > 0

    def test_corpus_word_doc_counts_error(self, corpus):
        with pytest.raises(ValueError):
            corpus.word_doc_counts(weighting="foo")

    def test_corpus_save_and_load(self, corpus, tmpdir):
        filepath = str(tmpdir.join("test_corpus_save_and_load.pkl"))
        corpus.save(filepath)
        loaded_corpus = Corpus.load("en", filepath)
        assert isinstance(loaded_corpus, Corpus)
        assert len(loaded_corpus) == len(corpus)
        assert loaded_corpus.spacy_lang.meta == corpus.spacy_lang.meta
        assert loaded_corpus.spacy_lang.pipe_names == corpus.spacy_lang.pipe_names
        assert corpus[0].user_data == loaded_corpus[0].user_data
