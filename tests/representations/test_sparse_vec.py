import pytest
import scipy.sparse as sp

from textacy import extract, representations


@pytest.fixture(scope="module")
def tokenized_docs(lang_en, text_lines_en):
    docs = list(lang_en.pipe(text_lines_en))
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
