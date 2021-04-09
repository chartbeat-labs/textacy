import numpy as np
import pytest
import scipy.sparse as sp
from spacy.tokens import Doc, Span

import textacy.datasets
from textacy import Corpus
from textacy import extract, preprocessing
from textacy import load_spacy_lang, make_spacy_doc
from textacy.extract import keyterms as kt
from textacy.text_stats import TextStats
from textacy.tm import TopicModel
from textacy.representations import Vectorizer

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
    spacy_lang = load_spacy_lang("en_core_web_sm")
    return make_spacy_doc(text, lang=spacy_lang)


@pytest.fixture(scope="module")
def corpus():
    records = DATASET.records(speaker_name={"Bernie Sanders"}, limit=10)
    corpus = Corpus("en_core_web_sm", data=records)
    return corpus


def test_streaming_functionality(corpus):
    assert isinstance(DATASET, textacy.datasets.base.Dataset)
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
            (term.text for term in doc._.extract_terms(ngs=1, ents=True, ncs=True))
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
    preprocessed_text = preprocessing.normalize.whitespace(text)
    preprocessed_text = preprocessing.remove.punctuation(text)
    preprocessed_text = preprocessed_text.lower()
    assert all(char.islower() for char in preprocessed_text if char.isalpha())
    assert all(char.isalnum() or char.isspace() for char in preprocessed_text)
    keyword = "America"
    kwics = extract.keyword_in_context(text, keyword, window_width=35)
    for pre, kw, post in kwics:
        assert kw == keyword
        assert isinstance(pre, str)
        assert isinstance(post, str)


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

    regex_matches = list(extract.regex_matches(doc, "Mr\. Speaker"))[:10]
    for match in regex_matches:
        assert isinstance(match, Span)

    stmts = list(extract.semistructured_statements(doc, entity="I", cue="be"))[:10]
    for stmt in stmts:
        assert isinstance(stmt, list)
        assert isinstance(stmt[0], str)
        assert len(stmt) == 3

    kts = kt.textrank(doc, topn=10)
    for keyterm in kts:
        assert isinstance(keyterm, tuple)
        assert isinstance(keyterm[0], str)
        assert isinstance(keyterm[1], float)
        assert keyterm[1] > 0.0


def test_text_stats_functionality(doc):
    ts = TextStats(doc)

    for attr in ["n_words", "n_syllables", "n_chars"]:
        assert hasattr(ts, attr)
        assert isinstance(getattr(ts, attr), int)

    for attr in ["entropy", "flesch_kincaid_grade_level", "flesch_reading_ease", "lix"]:
        assert hasattr(ts, attr)
        assert isinstance(getattr(ts, attr), float)
