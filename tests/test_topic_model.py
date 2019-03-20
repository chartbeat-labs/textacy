# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals

import numpy as np
import pytest
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from textacy import Corpus, TopicModel, Vectorizer


@pytest.fixture(scope="module")
def term_lists():
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
    term_lists_ = [
        doc.to_terms_list(ngrams=1, named_entities=False, as_strings=True)
        for doc in corpus
    ]
    return term_lists_


@pytest.fixture(scope="module")
def vectorizer():
    vectorizer_ = Vectorizer(
        tf_type="linear",
        apply_idf=True,
        idf_type="smooth",
        norm=None,
        min_df=1,
        max_df=1.0,
        max_n_terms=None,
    )
    return vectorizer_


@pytest.fixture(scope="module")
def doc_term_matrix(term_lists, vectorizer):
    doc_term_matrix_ = vectorizer.fit_transform(term_lists)
    return doc_term_matrix_


@pytest.fixture(scope="module")
def model(doc_term_matrix):
    model_ = TopicModel("nmf", n_topics=5)
    model_.fit(doc_term_matrix)
    return model_


def test_n_topics():
    for model in ["nmf", "lda", "lsa"]:
        assert TopicModel(model, n_topics=20).n_topics == 20


def test_init_model():
    expecteds = (NMF, LatentDirichletAllocation, TruncatedSVD)
    models = ["nmf", "lda", "lsa"]
    for model, expected in zip(models, expecteds):
        assert isinstance(TopicModel(model).model, expected)


def test_save_load(tmpdir, model):
    filename = str(tmpdir.join("model.pkl"))
    expected = model.model.components_
    model.save(filename)
    tmp_model = TopicModel.load(filename)
    observed = tmp_model.model.components_
    assert observed.shape == expected.shape
    assert np.equal(observed, expected).all()


def test_transform(doc_term_matrix, model):
    expected = (doc_term_matrix.shape[0], model.n_topics)
    observed = model.transform(doc_term_matrix).shape
    assert observed == expected


@pytest.mark.skip(reason="this sometimes fails randomly, reason unclear...")
def test_get_doc_topic_matrix(doc_term_matrix, model):
    expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    observed = model.get_doc_topic_matrix(doc_term_matrix, normalize=True).sum(axis=1)
    assert np.equal(observed, expected).all()


def test_get_doc_topic_matrix_nonnormalized(doc_term_matrix, model):
    expected = model.transform(doc_term_matrix)
    observed = model.get_doc_topic_matrix(doc_term_matrix, normalize=False)
    assert np.equal(observed, expected).all()


def test_top_topic_terms_topics(vectorizer, model):
    assert (
        len(list(model.top_topic_terms(vectorizer.id_to_term, topics=-1)))
        == model.n_topics
    )
    assert len(list(model.top_topic_terms(vectorizer.id_to_term, topics=0))) == 1
    observed = [
        topic_idx
        for topic_idx, _ in model.top_topic_terms(
            vectorizer.id_to_term, topics=(1, 2, 3)
        )
    ]
    expected = [1, 2, 3]
    assert observed == expected


def test_top_topic_terms_top_n(vectorizer, model):
    assert (
        len(
            list(model.top_topic_terms(vectorizer.id_to_term, topics=0, top_n=10))[0][1]
        )
        == 10
    )
    assert (
        len(list(model.top_topic_terms(vectorizer.id_to_term, topics=0, top_n=5))[0][1])
        == 5
    )


def test_top_topic_terms_weights(vectorizer, model):
    observed = list(
        model.top_topic_terms(vectorizer.id_to_term, topics=-1, top_n=10, weights=True)
    )
    assert isinstance(observed[0][1][0], tuple)
    for topic_idx, term_weights in observed:
        for i in range(len(term_weights) - 1):
            assert term_weights[i][1] >= term_weights[i + 1][1]
