"""
Sparse Vectors
--------------

:mod:`textacy.representations.sparse_vec`: Transform a collection of tokenized docs
into a doc-term matrix of shape (# docs, # unique terms) or a group-term matrix
of shape (# unique groups, # unique terms), with various ways to filter/limit
included terms and flexible weighting/normalization schemes for their values.

Intended primarily as a simpler- and higher-level API for sparse vectorization of docs.
"""
from typing import Iterable, Literal, Optional

import scipy.sparse as sp

from . import vectorizers


def build_doc_term_matrix(
    tokenized_docs: Iterable[Iterable[str]],
    *,
    tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
    idf_type: Optional[Literal["standard", "smooth", "bm25"]] = None,
    dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
    **kwargs,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """
    Transform one or more tokenized documents into a document-term matrix
    of shape (# docs, # unique terms), with flexible weighting/normalization of values.

    Args:
        tokenized_docs: A sequence of tokenized documents, where each is a sequence
            of term strings. For example::

                >>> ([tok.lemma_ for tok in spacy_doc]
                ...  for spacy_doc in spacy_docs)
                >>> ((ne.text for ne in extract.entities(doc))
                ...  for doc in corpus)

        tf_type: Type of term frequency (tf) to use for weights' local component:

            - "linear": tf (tfs are already linear, so left as-is)
            - "sqrt": tf => sqrt(tf)
            - "log": tf => log(tf) + 1
            - "binary": tf => 1

        idf_type: Type of inverse document frequency (idf) to use for weights'
            global component:

            - "standard": idf = log(n_docs / df) + 1.0
            - "smooth": idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - "bm25": idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.
            - None: no global weighting is applied to local term weights.

        dl_type: Type of document-length scaling to use for weights'
            normalization component:

            - "linear": dl (dls are already linear, so left as-is)
            - "sqrt": dl => sqrt(dl)
            - "log": dl => log(dl)
            - None: no normalization is applied to local (+global?) weights

        **kwargs: Passed directly into vectorizer class

    Returns:
        Document-term matrix as a sparse row matrix, and
        the corresponding mapping of term strings to integer ids (column indexes).

    Note:
        If you need to transform other sequences of tokenized documents in the same way,
        or if you need more access to the underlying vectorization process,
        consider using :class:`textacy.representations.vectorizers.Vectorizer` directly.

    See Also:
        - :class:`textacy.representations.vectorizers.Vectorizer`
        - :class:`scipy.sparse.csr_matrix`

    Reference:
        https://en.wikipedia.org/wiki/Document-term_matrix
    """
    vectorizer = vectorizers.Vectorizer(
        tf_type=tf_type, idf_type=idf_type, dl_type=dl_type, **kwargs
    )
    doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
    return (doc_term_matrix, vectorizer.vocabulary_terms)


def build_grp_term_matrix(
    tokenized_docs: Iterable[Iterable[str]],
    grps: Iterable[str],
    *,
    tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
    idf_type: Optional[Literal["standard", "smooth", "bm25"]] = None,
    dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
    **kwargs,
) -> tuple[sp.csr_matrix, dict[str, int], dict[str, int]]:
    """
    Transform one or more tokenized documents into a group-term matrix
    of shape (# unique groups, # unique terms),
    with flexible weighting/normalization of values.

    This is an extension of typical document-term matrix vectorization, where
    terms are grouped by the documents in which they co-occur. It allows for
    customized grouping, such as by a shared author or publication year, that
    may span multiple documents, without forcing users to merge those documents
    themselves.

    Args:
        tokenized_docs: A sequence of tokenized documents, where each is a sequence
            of term strings. For example::

                >>> ([tok.lemma_ for tok in spacy_doc]
                ...  for spacy_doc in spacy_docs)
                >>> ((ne.text for ne in extract.entities(doc))
                ...  for doc in corpus)

        grps: Sequence of group names by which the terms in ``tokenized_docs``
            are aggregated, where the first item in ``grps`` corresponds to
            the first item in ``tokenized_docs``, and so on.
        tf_type: Type of term frequency (tf) to use for weights' local component:

            - "linear": tf (tfs are already linear, so left as-is)
            - "sqrt": tf => sqrt(tf)
            - "log": tf => log(tf) + 1
            - "binary": tf => 1

        idf_type: Type of inverse document frequency (idf) to use for weights'
            global component:

            - "standard": idf = log(n_docs / df) + 1.0
            - "smooth": idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - "bm25": idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.
            - None: no global weighting is applied to local term weights.

        dl_type: Type of document-length scaling to use for weights'
            normalization component:

            - "linear": dl (dls are already linear, so left as-is)
            - "sqrt": dl => sqrt(dl)
            - "log": dl => log(dl)
            - None: no normalization is applied to local (+global?) weights

        **kwargs: Passed directly into vectorizer class

    Returns:
        Group-term matrix as a sparse row matrix, and
        the corresponding mapping of term strings to integer ids (column indexes), and
        the corresponding mapping of group strings to integer ids (row indexes).

    Note:
        If you need to transform other sequences of tokenized documents in the same way,
        or if you need more access to the underlying vectorization process, consider
        using :class:`textacy.representations.vectorizers.GroupVectorizer` directly.

    See Also:
        - :class:`textacy.representations.vectorizers.GroupVectorizer`
        - :class:`scipy.sparse.csr_matrix`

    Reference:
        https://en.wikipedia.org/wiki/Document-term_matrix
    """
    vectorizer = vectorizers.GroupVectorizer(
        tf_type=tf_type, idf_type=idf_type, dl_type=dl_type, **kwargs
    )
    grp_term_matrix = vectorizer.fit_transform(tokenized_docs, grps)
    return (grp_term_matrix, vectorizer.vocabulary_terms, vectorizer.vocabulary_grps)
