from __future__ import absolute_import, unicode_literals

from operator import itemgetter

import numpy as np
import pytest
import scipy.sparse as sp
from spacy.tokens import Doc, Span

import textacy.datasets
import textacy.ke
from textacy import Corpus, TextStats
from textacy import (
    cache,
    compat,
    constants,
    extract,
    io,
    preprocessing,
    text_utils,
)
from textacy.doc import make_spacy_doc
from textacy.tm import TopicModel
from textacy.vsm import Vectorizer

DATASET = textacy.datasets.CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def text():
    return next(DATASET.texts(speaker_name={"Bernie Sanders"}, limit=1)).strip()


@pytest.fixture(scope="module")
def doc(text):
    spacy_lang = cache.load_spacy_lang("en")
    return make_spacy_doc(text, lang=spacy_lang)


@pytest.fixture(scope="module")
def corpus():
    records = DATASET.records(speaker_name={"Bernie Sanders"}, limit=10)
    corpus = Corpus("en", data=records)
    return corpus


def test_streaming_functionality(corpus):
    assert isinstance(DATASET, textacy.datasets.dataset.Dataset)
    assert isinstance(corpus, Corpus)


def test_vectorization_and_topic_modeling_functionality(corpus):
    n_topics = 10
    top_n = 10
    vectorizer = Vectorizer(
        tf_type="linear",
        apply_idf=True,
        idf_type="smooth",
        norm=None,
        min_df=2,
        max_df=0.95,
    )
    doc_term_matrix = vectorizer.fit_transform(
        (
            doc._.to_terms_list(ngrams=1, entities=True, as_strings=True)
            for doc in corpus
        )
    )
    model = TopicModel("nmf", n_topics=n_topics)
    model.fit(doc_term_matrix)
    doc_topic_matrix = model.transform(doc_term_matrix)
    assert isinstance(doc_term_matrix, sp.csr_matrix)
    assert isinstance(doc_topic_matrix, np.ndarray)
    assert doc_topic_matrix.shape[1] == n_topics
    for topic_idx, top_terms in model.top_topic_terms(
        vectorizer.id_to_term, top_n=top_n
    ):
        assert isinstance(topic_idx, int)
        assert len(top_terms) == top_n


def test_corpus_functionality(corpus):
    assert isinstance(corpus[0], Doc)
    assert list(
        corpus.get(lambda doc: doc._.meta["speaker_name"] == "Bernie Sanders")
    )


def test_plaintext_functionality(text):
    preprocessed_text = preprocessing.normalize_whitespace(text)
    preprocessed_text = preprocessing.remove_punctuation(text)
    preprocessed_text = preprocessed_text.lower()
    assert all(char.islower() for char in preprocessed_text if char.isalpha())
    assert all(char.isalnum() or char.isspace() for char in preprocessed_text)
    keyword = "America"
    kwics = text_utils.keyword_in_context(
        text, keyword, window_width=35, print_only=False
    )
    for pre, kw, post in kwics:
        assert kw == keyword
        assert isinstance(pre, compat.unicode_)
        assert isinstance(post, compat.unicode_)


def test_extract_functionality(doc):
    bigrams = list(
        extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=False)
    )[:10]
    for bigram in bigrams:
        assert isinstance(bigram, Span)
        assert len(bigram) == 2

    trigrams = list(
        extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, min_freq=2)
    )[:10]
    for trigram in trigrams:
        assert isinstance(trigram, Span)
        assert len(trigram) == 3

    nes = list(
        extract.entities(doc, drop_determiners=False, exclude_types="numeric")
    )[:10]
    for ne in nes:
        assert isinstance(ne, Span)
        assert ne.label_
        assert ne.label_ != "QUANTITY"

    pos_regex_matches = list(
        extract.pos_regex_matches(doc, constants.POS_REGEX_PATTERNS["en"]["NP"])
    )[:10]
    for match in pos_regex_matches:
        assert isinstance(match, Span)

    stmts = list(extract.semistructured_statements(doc, "I", cue="be"))[:10]
    for stmt in stmts:
        assert isinstance(stmt, list)
        assert isinstance(stmt[0], compat.unicode_)
        assert len(stmt) == 3

    kts = textacy.ke.textrank(doc, topn=10)
    for keyterm in kts:
        assert isinstance(keyterm, tuple)
        assert isinstance(keyterm[0], compat.unicode_)
        assert isinstance(keyterm[1], float)
        assert keyterm[1] > 0.0


def test_text_stats_functionality(doc):
    ts = TextStats(doc)

    assert isinstance(ts.n_words, int)
    assert isinstance(ts.flesch_kincaid_grade_level, float)

    basic_counts = ts.basic_counts
    assert isinstance(basic_counts, dict)
    for field in ("n_chars", "n_words", "n_sents"):
        assert isinstance(basic_counts.get(field), int)

    readability_stats = ts.readability_stats
    assert isinstance(readability_stats, dict)
    for field in (
        "flesch_kincaid_grade_level",
        "automated_readability_index",
        "wiener_sachtextformel",
    ):
        assert isinstance(readability_stats.get(field), float)
