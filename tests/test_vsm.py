# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest
from scipy.sparse import coo_matrix

from textacy import Corpus
from textacy import compat, vsm


@pytest.fixture(scope='module')
def vectorizer_and_dtm():
    texts = ["Mary had a little lamb. Its fleece was white as snow.",
             "Everywhere that Mary went the lamb was sure to go.",
             "It followed her to school one day, which was against the rule.",
             "It made the children laugh and play to see a lamb at school.",
             "And so the teacher turned it out, but still it lingered near.",
             "It waited patiently about until Mary did appear.",
             "Why does the lamb love Mary so? The eager children cry.",
             "Mary loves the lamb, you know, the teacher did reply."]
    corpus = Corpus('en', texts=texts)
    term_lists = [
        doc.to_terms_list(ngrams=1, named_entities=False, as_strings=True)
        for doc in corpus]
    vectorizer = vsm.Vectorizer(
        weighting='tf', normalize=False, sublinear_tf=False, smooth_idf=True,
        min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None)
    doc_term_matrix = vectorizer.fit_transform(term_lists)
    return vectorizer, doc_term_matrix


@pytest.fixture(scope='module')
def lamb_and_child_idxs(vectorizer_and_dtm):
    vectorizer, _ = vectorizer_and_dtm
    idx_lamb = [
        id_ for term, id_ in vectorizer.vocabulary_terms.items()
        if term == 'lamb'][0]
    idx_child = [
        id_ for term, id_ in vectorizer.vocabulary_terms.items()
        if term == 'child'][0]
    return idx_lamb, idx_child


def test_vectorizer_feature_names(vectorizer_and_dtm):
    vectorizer, _ = vectorizer_and_dtm
    assert isinstance(vectorizer.terms_list, list)
    assert isinstance(vectorizer.terms_list[0], compat.unicode_)
    assert len(vectorizer.terms_list) == len(vectorizer.vocabulary_terms)


def test_vectorizer_bad_init_params():
    bad_init_params = (
        {'min_df': -1},
        {'max_df': -1},
        {'max_n_terms': -1},
        {'min_ic': -1.0},
        {'vocabulary_terms': 'foo bar bat baz'},
        )
    for bad_init_param in bad_init_params:
        with pytest.raises(ValueError):
            _ = vsm.Vectorizer(**bad_init_param)


def test_get_term_freqs(vectorizer_and_dtm, lamb_and_child_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_child = lamb_and_child_idxs
    term_freqs = vsm.get_term_freqs(doc_term_matrix, normalized=False)
    assert len(term_freqs) == doc_term_matrix.shape[1]
    assert term_freqs.min() == 1
    assert term_freqs.max() == 5
    assert term_freqs[idx_lamb] == 5
    assert term_freqs[idx_child] == 2


def test_get_term_freqs_normalized(vectorizer_and_dtm, lamb_and_child_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_child = lamb_and_child_idxs
    term_freqs = vsm.get_term_freqs(doc_term_matrix, normalized=True)
    assert len(term_freqs) == doc_term_matrix.shape[1]
    assert term_freqs.max() == pytest.approx(0.15625, abs=1e-3)
    assert term_freqs.min() == pytest.approx(0.03125, abs=1e-3)
    assert term_freqs[idx_lamb] == pytest.approx(0.15625, abs=1e-3)
    assert term_freqs[idx_child] == pytest.approx(0.06250, abs=1e-3)


def test_get_term_freqs_exception():
    with pytest.raises(ValueError):
        _ = vsm.get_term_freqs(coo_matrix((1, 1)).tocsr())


def test_get_doc_freqs(vectorizer_and_dtm, lamb_and_child_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_child = lamb_and_child_idxs
    doc_freqs = vsm.get_doc_freqs(doc_term_matrix, normalized=False)
    assert len(doc_freqs) == doc_term_matrix.shape[1]
    assert doc_freqs.max() == 5
    assert doc_freqs.min() == 1
    assert doc_freqs[idx_lamb] == 5
    assert doc_freqs[idx_child] == 2


def test_get_doc_freqs_normalized(vectorizer_and_dtm, lamb_and_child_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_child = lamb_and_child_idxs
    doc_freqs = vsm.get_doc_freqs(doc_term_matrix, normalized=True)
    assert len(doc_freqs) == doc_term_matrix.shape[1]
    assert doc_freqs.max() == pytest.approx(0.625, rel=1e-3)
    assert doc_freqs.min() == pytest.approx(0.125, rel=1e-3)
    assert doc_freqs[idx_lamb] == pytest.approx(0.625, rel=1e-3)
    assert doc_freqs[idx_child] == pytest.approx(0.250, rel=1e-3)


def test_get_doc_freqs_exception():
    with pytest.raises(ValueError):
        _ = vsm.get_doc_freqs(coo_matrix((1, 1)).tocsr())


def test_get_information_content(vectorizer_and_dtm, lamb_and_child_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_child = lamb_and_child_idxs
    ics = vsm.get_information_content(doc_term_matrix)
    assert len(ics) == doc_term_matrix.shape[1]
    assert ics.max() == pytest.approx(1.0, rel=1e-3)
    assert ics.min() == pytest.approx(0.54356, rel=1e-3)
    assert ics[idx_lamb] == pytest.approx(0.95443, rel=1e-3)
    assert ics[idx_child] == pytest.approx(0.81127, rel=1e-3)


def test_filter_terms_by_df_identity(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix, vectorizer.vocabulary_terms,
        max_df=1.0, min_df=1, max_n_terms=None)
    assert dtm.shape == doc_term_matrix.shape
    assert vocab == vectorizer.vocabulary_terms


def test_filter_terms_by_df_max_n_terms(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix, vectorizer.vocabulary_terms,
        max_df=1.0, min_df=1, max_n_terms=2)
    assert dtm.shape == (8, 2)
    assert sorted(vocab.keys()) == ['lamb', 'mary']


def test_filter_terms_by_df_min_df(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix, vectorizer.vocabulary_terms,
        max_df=1.0, min_df=2, max_n_terms=None)
    assert dtm.shape == (8, 7)
    assert sorted(vocab.keys()) == ['-PRON-', 'child', 'lamb', 'love', 'mary', 'school', 'teacher']


def test_filter_terms_by_df_exception(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    with pytest.raises(ValueError):
        _ = vsm.filter_terms_by_df(doc_term_matrix, vectorizer.vocabulary_terms,
                                   max_df=1.0, min_df=6, max_n_terms=None)


def test_filter_terms_by_ic_identity(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_ic(
        doc_term_matrix, vectorizer.vocabulary_terms,
        min_ic=0.0, max_n_terms=None)
    assert dtm.shape == doc_term_matrix.shape
    assert vocab == vectorizer.vocabulary_terms


def test_filter_terms_by_ic_max_n_terms(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_ic(
        doc_term_matrix, vectorizer.vocabulary_terms,
        min_ic=0.0, max_n_terms=3)
    assert dtm.shape == (8, 3)
    assert len(vocab) == 3
