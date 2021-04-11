import collections
import itertools
import re

import pytest
from spacy.tokens import Span

import textacy
import textacy.datasets
import textacy.extract
import textacy.preprocessing


pytestmark = pytest.mark.skipif(
    textacy.datasets.CapitolWords().filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.fixture(scope="module")
def dataset():
    dataset = textacy.datasets.CapitolWords()
    return dataset


@pytest.fixture(scope="module")
def record(dataset):
    record = next(dataset.records(limit=1))
    return record


@pytest.fixture(scope="module")
def doc(record):
    doc = textacy.make_spacy_doc(record, lang="en_core_web_sm")
    return doc


@pytest.fixture(scope="module")
def corpus(dataset):
    corpus = textacy.Corpus("en_core_web_sm", data=dataset.records(limit=10))
    return corpus


def test_dataset(dataset):
    assert hasattr(dataset, "info")
    assert isinstance(dataset.info, dict)
    assert hasattr(dataset, "download")


def test_dataset_record(record):
    assert isinstance(record, tuple)
    assert len(record) == 2
    assert isinstance(record[0], str) and isinstance(record[1], dict)
    assert hasattr(record, "text") and hasattr(record, "meta")


def test_extract_kwic(record):
    results = list(
        textacy.extract.keyword_in_context(
            record.text, "work(ing|ers?)", window_width=35
        )
    )
    assert results
    assert len(results) == 4
    assert all(isinstance(result, tuple) for result in results)


def test_preprocessing(record):
    result = textacy.preprocessing.replace.numbers(record.text)
    assert isinstance(result, str)
    assert not any(re.findall(r"\d+", result))


def test_preprocessing_pipeline(record):
    preproc = textacy.preprocessing.make_pipeline(
        textacy.preprocessing.normalize.unicode,
        textacy.preprocessing.normalize.quotation_marks,
        textacy.preprocessing.normalize.whitespace,
    )
    result = preproc(record.text)
    assert result and isinstance(result, str)


def test_doc_extensions(doc):
    assert all(hasattr(doc._, attr) for attr in ["preview", "meta"])
    assert isinstance(doc._.preview, str)
    assert isinstance(doc._.meta, dict)


def test_extract_token_matches(doc):
    patterns = [
        {"POS": {"IN": ["ADJ", "DET"]}, "OP": "+"}, {"ORTH": {"REGEX": "workers?"}}
    ]
    results = list(textacy.extract.token_matches(doc, patterns))
    assert results and len(results) == 2
    assert all(isinstance(result, Span) for result in results)


def test_corpus_metadata(corpus):
    result = corpus.agg_metadata("congress", min)
    assert result and isinstance(result, int)
    min_dt = corpus.agg_metadata("date", min)
    max_dt = corpus.agg_metadata("date", max)
    assert min_dt < max_dt
    result = corpus.agg_metadata("speaker_name", collections.Counter)
    assert result and isinstance(result, collections.Counter)


def test_corpus_extract_token_matches(corpus):
    patterns = [
        {"POS": {"IN": ["ADJ", "DET"]}, "OP": "+"}, {"ORTH": {"REGEX": "workers?"}}
    ]
    matches = itertools.chain.from_iterable(
        textacy.extract.token_matches(doc, patterns) for doc in corpus
    )
    result = collections.Counter(match.lemma_ for match in matches).most_common(10)
    assert all(isinstance(key, str) for key, _ in result)
    assert all(isinstance(val, int) for _, val in result)


def test_doc_extract_keyterms(doc):
    result = doc._.extract_keyterms(
        "textrank", normalize="lemma", window_size=10, edge_weighting="count", topn=10
    )
    assert result and isinstance(result, list)
    assert all(isinstance(key, str) for key, _ in result)
    assert all(isinstance(val, float) for _, val in result)
    assert (
        sorted([val for _, val in result], reverse=True) == [val for _, val in result]
    )
