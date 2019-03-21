# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest
import numpy as np
from scipy.sparse import coo_matrix

from textacy import Corpus
from textacy import compat, vsm


@pytest.fixture(scope="module")
def tokenized_docs():
    texts = [
        "Mary had a little lamb. Its fleece was white as snow.",
        "Everywhere that Mary went the lamb was sure to go.",
        "It followed her to school one day, which was against the rule.",
        "It made the children laugh and play to see a lamb at school.",
        "And so the teacher turned it out, but still it lingered near.",
        "It waited patiently about until Mary did appear.",
        "Why does the lamb love Mary so? The eager children cry.",
        "Mary loves the lamb, you know, the teacher did reply.",
    ]
    corpus = Corpus("en", texts=texts)
    tokenized_docs = [
        list(doc.to_terms_list(ngrams=1, named_entities=False, normalize="lower", as_strings=True))
        for doc in corpus
    ]
    return tokenized_docs


@pytest.fixture(scope="module")
def groups():
    return ["a", "b", "c", "a", "b", "c", "a", "b"]


@pytest.fixture(scope="module")
def vectorizer_and_dtm(tokenized_docs):
    vectorizer = vsm.Vectorizer(
        tf_type="linear",
        idf_type="smooth",
        norm=None,
        min_df=1,
        max_df=1.0,
        max_n_terms=None,
    )
    doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
    return vectorizer, doc_term_matrix


@pytest.fixture(scope="module")
def grp_vectorizer_and_gtm(tokenized_docs, groups):
    grp_vectorizer = vsm.GroupVectorizer(
        tf_type="linear",
        idf_type="smooth",
        norm=None,
        min_df=1,
        max_df=1.0,
        max_n_terms=None,
    )
    grp_term_matrix = grp_vectorizer.fit_transform(tokenized_docs, groups)
    return grp_vectorizer, grp_term_matrix


@pytest.fixture(scope="module")
def grp_vectorizer_and_gtm_2(tokenized_docs, groups):
    grp_vectorizer = vsm.GroupVectorizer(
        tf_type="bm25",
        idf_type="smooth",
        norm=None,
        apply_dl=True,
        min_df=1,
        max_df=1.0,
        max_n_terms=None,
    )
    grp_term_matrix = grp_vectorizer.fit_transform(tokenized_docs, groups)
    return grp_vectorizer, grp_term_matrix


@pytest.fixture(scope="module")
def lamb_and_children_idxs(vectorizer_and_dtm):
    vec, _ = vectorizer_and_dtm
    idx_lamb = [id_ for term, id_ in vec.vocabulary_terms.items() if term == "lamb"][0]
    idx_children = [id_ for term, id_ in vec.vocabulary_terms.items() if term == "children"][0]
    return idx_lamb, idx_children


def test_vectorizer_weighting_combinations(tokenized_docs):
    init_params = [
        dict(tf_type="linear"),
        dict(tf_type="sqrt"),
        dict(tf_type="sqrt", apply_dl=True),
        dict(tf_type="sqrt", apply_dl=True, dl_type="sqrt"),
        dict(tf_type="linear", apply_idf=True),
        dict(tf_type="linear", apply_idf=True, idf_type="bm25"),
        dict(tf_type="linear", apply_idf=True, idf_type="standard", norm="l1"),
        dict(tf_type="linear", apply_idf=True, idf_type="standard", apply_dl=True),
        dict(tf_type="linear", apply_idf=True, idf_type="smooth", apply_dl=True, dl_type="log"),
        dict(tf_type="bm25", apply_idf=True, idf_type="bm25"),
        dict(tf_type="bm25", apply_idf=True, apply_dl=False),
        dict(tf_type="bm25", apply_idf=True, idf_type="bm25"),
        dict(tf_type="bm25", apply_idf=True, idf_type="smooth", norm="l2"),
    ]
    for ip in init_params:
        vectorizer = vsm.Vectorizer(**ip)
        doc_term_matrix = vectorizer.fit(tokenized_docs)
        vectorizer.weighting


def test_vectorizer_id_to_term(vectorizer_and_dtm):
    vectorizer, _ = vectorizer_and_dtm
    assert isinstance(vectorizer.id_to_term, dict)
    assert all(isinstance(key, int) for key in vectorizer.id_to_term)
    assert all(
        isinstance(val, compat.unicode_) for val in vectorizer.id_to_term.values()
    )
    assert len(vectorizer.id_to_term) == len(vectorizer.vocabulary_terms)


def test_vectorizer_terms_list(vectorizer_and_dtm):
    vectorizer, dtm = vectorizer_and_dtm
    assert isinstance(vectorizer.terms_list, list)
    assert isinstance(vectorizer.terms_list[0], compat.unicode_)
    assert len(vectorizer.terms_list) == len(vectorizer.vocabulary_terms)
    assert len(vectorizer.terms_list) == dtm.shape[1]
    assert vectorizer.terms_list == sorted(vectorizer.terms_list)


def test_grp_vectorizer_id_to_grp(grp_vectorizer_and_gtm):
    grp_vectorizer, _ = grp_vectorizer_and_gtm
    assert isinstance(grp_vectorizer.id_to_grp, dict)
    assert all(isinstance(key, int) for key in grp_vectorizer.id_to_grp)
    assert all(
        isinstance(val, compat.unicode_) for val in grp_vectorizer.id_to_grp.values()
    )
    assert len(grp_vectorizer.id_to_grp) == len(grp_vectorizer.vocabulary_grps)


def test_grp_vectorizer_terms_and_grp_list(grp_vectorizer_and_gtm):
    grp_vectorizer, gtm = grp_vectorizer_and_gtm
    assert isinstance(grp_vectorizer.terms_list, list)
    assert isinstance(grp_vectorizer.grps_list, list)
    assert isinstance(grp_vectorizer.terms_list[0], compat.unicode_)
    assert len(grp_vectorizer.terms_list) == len(grp_vectorizer.vocabulary_terms)
    assert len(grp_vectorizer.terms_list) == gtm.shape[1]
    assert grp_vectorizer.terms_list == sorted(grp_vectorizer.terms_list)
    assert len(grp_vectorizer.grps_list) == len(grp_vectorizer.vocabulary_grps)
    assert len(grp_vectorizer.grps_list) == gtm.shape[0]
    assert grp_vectorizer.grps_list == sorted(grp_vectorizer.grps_list)


def test_grp_vectorizer_2_fits(grp_vectorizer_and_gtm_2):
    grp_vectorizer, _ = grp_vectorizer_and_gtm_2


def test_vectorizer_fixed_vocab(tokenized_docs):
    vocabulary_terms = ["lamb", "snow", "school", "rule", "teacher"]
    vectorizer = vsm.Vectorizer(vocabulary_terms=vocabulary_terms)
    doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
    assert len(vectorizer.vocabulary_terms) == len(vocabulary_terms)
    assert doc_term_matrix.shape[1] == len(vocabulary_terms)
    assert sorted(vectorizer.terms_list) == sorted(vocabulary_terms)


def test_grp_vectorizer_fixed_vocab(tokenized_docs, groups):
    vocabulary_terms = ["lamb", "snow", "school", "rule", "teacher"]
    vocabulary_grps = ["a", "b"]
    grp_vectorizer = vsm.GroupVectorizer(
        vocabulary_terms=vocabulary_terms, vocabulary_grps=vocabulary_grps
    )
    grp_term_matrix = grp_vectorizer.fit_transform(tokenized_docs, groups)
    assert len(grp_vectorizer.vocabulary_terms) == len(vocabulary_terms)
    assert grp_term_matrix.shape[1] == len(vocabulary_terms)
    assert sorted(grp_vectorizer.terms_list) == sorted(vocabulary_terms)
    assert len(grp_vectorizer.vocabulary_grps) == len(vocabulary_grps)
    assert grp_term_matrix.shape[0] == len(vocabulary_grps)
    assert sorted(grp_vectorizer.grps_list) == sorted(vocabulary_grps)


def test_vectorizer_bad_init_params():
    bad_init_params = (
        {"min_df": -1},
        {"max_df": -1},
        {"max_n_terms": -1},
        {"vocabulary_terms": "foo bar bat baz"},
    )
    for bad_init_param in bad_init_params:
        with pytest.raises(ValueError):
            _ = vsm.Vectorizer(**bad_init_param)


def test_vectorizer_bad_transform(tokenized_docs):
    vectorizer = vsm.Vectorizer()
    with pytest.raises(ValueError):
        _ = vectorizer.transform(tokenized_docs)


def test_grp_vectorizer_bad_transform(tokenized_docs, groups):
    grp_vectorizer = vsm.GroupVectorizer()
    with pytest.raises(ValueError):
        _ = grp_vectorizer.transform(tokenized_docs, groups)


def test_get_term_freqs(vectorizer_and_dtm, lamb_and_children_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_children = lamb_and_children_idxs
    term_freqs = vsm.get_term_freqs(doc_term_matrix, type_="linear")
    assert len(term_freqs) == doc_term_matrix.shape[1]
    assert term_freqs.min() == 1
    assert term_freqs.max() == 5
    assert term_freqs[idx_lamb] == 5
    assert term_freqs[idx_children] == 2


def test_get_term_freqs_sublinear(vectorizer_and_dtm, lamb_and_children_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_children = lamb_and_children_idxs
    tfs = vsm.get_term_freqs(doc_term_matrix, type_="linear")
    tfs_sqrt = vsm.get_term_freqs(doc_term_matrix, type_="sqrt")
    tfs_log = vsm.get_term_freqs(doc_term_matrix, type_="log")
    assert len(tfs) == len(tfs_sqrt) == len(tfs_log) == doc_term_matrix.shape[1]
    assert tfs_log.max() == pytest.approx(2.60943, abs=1e-3)
    assert tfs_log.min() == pytest.approx(1.0, abs=1e-3)
    assert tfs_log[idx_lamb] == pytest.approx(2.60943, abs=1e-3)
    assert tfs_log[idx_children] == pytest.approx(1.69314, abs=1e-3)
    assert (tfs_sqrt == np.sqrt(tfs)).all()
    assert (tfs_log == np.log(tfs) + 1.0).all()


def test_get_term_freqs_exception():
    with pytest.raises(ValueError):
        _ = vsm.get_term_freqs(coo_matrix((1, 1)).tocsr())


def test_get_doc_freqs(vectorizer_and_dtm, lamb_and_children_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_children = lamb_and_children_idxs
    doc_freqs = vsm.get_doc_freqs(doc_term_matrix)
    assert len(doc_freqs) == doc_term_matrix.shape[1]
    assert doc_freqs.max() == 5
    assert doc_freqs.min() == 1
    assert doc_freqs[idx_lamb] == 5
    assert doc_freqs[idx_children] == 2


def test_get_doc_freqs_exception():
    with pytest.raises(ValueError):
        _ = vsm.get_doc_freqs(coo_matrix((1, 1)).tocsr())


def test_get_doc_lengths(tokenized_docs, vectorizer_and_dtm):
    _, doc_term_matrix = vectorizer_and_dtm
    dls = vsm.get_doc_lengths(doc_term_matrix, type_="linear")
    assert len(dls) == doc_term_matrix.shape[0]
    for dl, td in zip(dls, tokenized_docs):
        assert dl == len(td)


def test_get_doc_lengths_type(vectorizer_and_dtm):
    _, doc_term_matrix = vectorizer_and_dtm
    dls = vsm.get_doc_lengths(doc_term_matrix, type_="linear")
    dls_sqrt = vsm.get_doc_lengths(doc_term_matrix, type_="sqrt")
    dls_log = vsm.get_doc_lengths(doc_term_matrix, type_="log")
    assert len(dls) == len(dls_sqrt) == len(dls_log) == doc_term_matrix.shape[0]
    assert (dls_sqrt == np.sqrt(dls)).all()
    assert (dls_log == np.log(dls) + 1.0).all()


def test_get_doc_lengths_exception(vectorizer_and_dtm):
    _, doc_term_matrix = vectorizer_and_dtm
    with pytest.raises(ValueError):
        _ = vsm.get_doc_lengths(doc_term_matrix, type_="foo")


def test_get_information_content(vectorizer_and_dtm, lamb_and_children_idxs):
    _, doc_term_matrix = vectorizer_and_dtm
    idx_lamb, idx_children = lamb_and_children_idxs
    ics = vsm.get_information_content(doc_term_matrix)
    assert len(ics) == doc_term_matrix.shape[1]
    assert ((0.0 <= ics) & (ics <= 1.0)).all()
    assert ics.max() > ics.min()
    assert ics[idx_lamb] == pytest.approx(0.95443, rel=0.001)
    assert ics[idx_children] == pytest.approx(0.81127, rel=0.001)


def test_apply_idf_weighting(vectorizer_and_dtm):
    _, doc_term_matrix = vectorizer_and_dtm
    for type_ in ("standard", "smooth", "bm25"):
        reweighted_matrix = vsm.apply_idf_weighting(doc_term_matrix, type_=type_)
        assert reweighted_matrix.shape == doc_term_matrix.shape


def test_filter_terms_by_df_identity(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix,
        vectorizer.vocabulary_terms,
        max_df=1.0,
        min_df=1,
        max_n_terms=None,
    )
    assert dtm.shape == doc_term_matrix.shape
    assert vocab == vectorizer.vocabulary_terms


def test_filter_terms_by_df_max_n_terms(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix,
        vectorizer.vocabulary_terms,
        max_df=1.0,
        min_df=1,
        max_n_terms=2,
    )
    assert dtm.shape[0] == doc_term_matrix.shape[0]
    assert dtm.shape[1] < doc_term_matrix.shape[1]
    assert all(term in vocab for term in ("lamb", "mary"))
    assert not any(term in vocab for term in ("children", "school", "teacher"))


def test_filter_terms_by_df_min_df(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    dtm, vocab = vsm.filter_terms_by_df(
        doc_term_matrix,
        vectorizer.vocabulary_terms,
        max_df=1.0,
        min_df=2,
        max_n_terms=None,
    )
    assert dtm.shape[0] == doc_term_matrix.shape[0]
    assert dtm.shape[1] < doc_term_matrix.shape[1]
    assert all(term in vocab for term in ("children", "lamb", "mary", "school", "teacher"))


def test_filter_terms_by_df_exception(vectorizer_and_dtm):
    vectorizer, doc_term_matrix = vectorizer_and_dtm
    with pytest.raises(ValueError):
        _ = vsm.filter_terms_by_df(
            doc_term_matrix,
            vectorizer.vocabulary_terms,
            max_df=1.0,
            min_df=6,
            max_n_terms=None,
        )
