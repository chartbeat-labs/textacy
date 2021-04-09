import pytest
import scipy.sparse as sp

import textacy
from textacy import extract, representations


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
    nlp = textacy.load_spacy_lang("en_core_web_sm")
    docs = list(nlp.pipe(texts))
    tokenized_docs = [
        [term.text.lower() for term in extract.terms(doc, ngs=1)] for doc in docs
    ]
    return tokenized_docs


@pytest.fixture(scope="module")
def groups():
    return ["a", "b", "c", "a", "b", "c", "a", "b"]


def test_build_doc_term_matrix(tokenized_docs):
    result = representations.build_doc_term_matrix(tokenized_docs)
    assert isinstance(result, tuple) and len(result) == 2
    dtm, vocab_terms = result
    assert isinstance(dtm, sp.csr_matrix)
    assert isinstance(vocab_terms, dict)


def test_build_grp_term_matrix(tokenized_docs, groups):
    result = representations.build_grp_term_matrix(tokenized_docs, groups)
    assert isinstance(result, tuple) and len(result) == 3
    gtm, vocab_terms, vocab_grps = result
    assert isinstance(gtm, sp.csr_matrix)
    assert isinstance(vocab_terms, dict)
    assert isinstance(vocab_grps, dict)
