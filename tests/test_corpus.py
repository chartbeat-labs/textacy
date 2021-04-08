import collections

import numpy as np
import pytest
import spacy
from spacy.tokens import Doc

from textacy import Corpus
from textacy import load_spacy_lang
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def en_core_web_sm():
    return load_spacy_lang("en_core_web_sm")


@pytest.fixture(scope="module")
def langs():
    return ["en_core_web_sm", load_spacy_lang("en_core_web_sm")]


@pytest.fixture(scope="module")
def corpus(en_core_web_sm):
    return Corpus(en_core_web_sm, data=DATASET.records(limit=5))


class TestCorpusInit:

    def test_init_lang(self, langs):
        for lang in langs:
            assert isinstance(Corpus(lang), Corpus)

    @pytest.mark.parametrize("lang", [b"en_core_web_sm", None])
    def test_init_bad_lang(self, lang):
        with pytest.raises(TypeError):
            Corpus(lang)

    def test_init_texts(self, en_core_web_sm):
        limit = 3
        texts = list(DATASET.texts(limit=limit))
        corpus = Corpus(en_core_web_sm, data=texts)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(text == doc.text for text, doc in zip(texts, corpus))

    def test_init_records(self, en_core_web_sm):
        limit = 3
        records = list(DATASET.records(limit=limit))
        corpus = Corpus(en_core_web_sm, data=records)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(record[0] == doc.text for record, doc in zip(records, corpus))
        assert all(record[1] == doc._.meta for record, doc in zip(records, corpus))

    def test_init_docs(self, en_core_web_sm):
        limit = 3
        texts = DATASET.texts(limit=limit)
        docs = [en_core_web_sm(text) for text in texts]
        corpus = Corpus(en_core_web_sm, data=docs)
        assert len(corpus) == len(corpus.docs) == limit
        assert all(doc.vocab is corpus.spacy_lang.vocab for doc in corpus)
        assert all(doc1 is doc2 for doc1, doc2 in zip(docs, corpus))

    def test_init_no_parser(self):
        spacy_lang = load_spacy_lang("en_core_web_sm", disable=("parser",))
        corpus = Corpus(spacy_lang, data=(spacy_lang("This is a sentence in a doc."),))
        assert len(corpus) == 1
        assert corpus.n_sents == 0


class TestCorpusDunder:

    def test_str(self, corpus):
        cstr = str(corpus)
        assert cstr.startswith("Corpus")
        assert all(f"{n}" in cstr for n in [corpus.n_docs, corpus.n_tokens])

    def test_len(self, corpus):
        assert isinstance(len(corpus), int)
        assert len(corpus) == len(corpus.docs) == corpus.n_docs

    def test_iter(self, corpus):
        assert isinstance(corpus, collections.abc.Iterable)

    def test_getitem(self, corpus):
        assert isinstance(corpus[0], Doc)
        assert all(isinstance(doc, Doc) for doc in corpus[0:2])

    def test_delitem(self, corpus):
        del corpus[-1]
        del corpus[-1:]
        with pytest.raises(TypeError):
            del corpus["foo"]


class TestCorpusProperties:

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


class TestCorpusMethods:

    def test_add(self, corpus, en_core_web_sm):
        datas = (
            "This is an english sentence.",
            ("This is an english sentence.", {"foo": "bar"}),
            en_core_web_sm("This is an english sentence."),
            ["This is one sentence.", "This is another sentence."],
            [("This is sentence #1.", {"foo": "bar"}), ("This is sentence #2.", {"bat": "baz"})],
            [en_core_web_sm("This is sentence #1"), en_core_web_sm("This is sentence #2")],
        )
        n_docs = corpus.n_docs
        for data in datas:
            corpus.add(data)
            assert corpus.n_docs > n_docs
            n_docs = corpus.n_docs

    @pytest.mark.xfail(reason="there seems to be bug in spacy for nprocess > 1 ...")
    def test_add_nprocess(self, corpus):
        datas = (
            ["This is one sentence.", "This is another sentence."],
            [("This is sentence #1.", {"foo": "bar"}), ("This is sentence #2.", {"bat": "baz"})],
        )
        n_docs = corpus.n_docs
        for data in datas:
            corpus.add(data, n_process=2)
            assert corpus.n_docs > n_docs
            n_docs = corpus.n_docs

    def test_add_typeerror(self, corpus):
        datas = (
            b"This is a byte string.",
            [b"This is a byte string.", b"This is another byte string."],
        )
        for data in datas:
            with pytest.raises(TypeError):
                corpus.add(data)

    def test_get(self, corpus):
        match_funcs = (
            lambda doc: True,
            lambda doc: doc._.meta.get("speaker_name") == "Bernie Sanders",
        )
        for match_func in match_funcs:
            assert len(list(corpus.get(match_func))) > 0
            assert len(list(corpus.get(match_func, limit=1))) == 1

    def test_remove(self, corpus):
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

    def test_word_counts(self, corpus):
        abs_counts = corpus.word_counts(weighting="count", by="lower")
        rel_counts = corpus.word_counts(weighting="freq", by="lower")
        assert isinstance(abs_counts, dict)
        assert all(isinstance(count, int) for count in abs_counts.values())
        assert min(abs_counts.values()) > 0
        assert isinstance(rel_counts, dict)
        assert all(isinstance(count, float) for count in rel_counts.values())
        assert min(rel_counts.values()) > 0 and max(rel_counts.values()) <= 1

    def test_word_counts_error(self, corpus):
        with pytest.raises(ValueError):
            corpus.word_counts(weighting="foo")

    def test_word_doc_counts(self, corpus):
        abs_counts = corpus.word_doc_counts(weighting="count", by="lower")
        rel_counts = corpus.word_doc_counts(weighting="freq", by="lower")
        inv_counts = corpus.word_doc_counts(weighting="idf", by="lower")
        assert isinstance(abs_counts, dict)
        assert all(isinstance(count, int) for count in abs_counts.values())
        assert min(abs_counts.values()) > 0
        assert isinstance(rel_counts, dict)
        assert all(isinstance(count, float) for count in rel_counts.values())
        assert min(rel_counts.values()) > 0 and max(rel_counts.values()) <= 1
        assert isinstance(inv_counts, dict)
        assert min(inv_counts.values()) > 0

    def test_word_doc_counts_error(self, corpus):
        with pytest.raises(ValueError):
            corpus.word_doc_counts(weighting="foo")

    def test_corpus_save_and_load(self, corpus, tmpdir, en_core_web_sm):
        filepath = str(tmpdir.join("test_corpus_save_and_load.bin"))
        corpus.save(filepath)
        loaded_corpus = Corpus.load(en_core_web_sm, filepath)
        assert isinstance(loaded_corpus, Corpus)
        assert len(loaded_corpus) == len(corpus)
        assert loaded_corpus.spacy_lang.meta == corpus.spacy_lang.meta
        assert loaded_corpus.spacy_lang.pipe_names == corpus.spacy_lang.pipe_names
        assert corpus[0].user_data == loaded_corpus[0].user_data
