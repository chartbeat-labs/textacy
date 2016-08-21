"""
Represent a collection of spacy-processed texts as a document-term matrix of shape
(# docs, # unique terms), with a variety of filtering, normalization, and term
weighting schemes for the values.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import binarize as binarize_mat
from sklearn.preprocessing import normalize as normalize_mat
from spacy.strings import StringStore


def doc_term_matrix(terms_lists, weighting='tf',
                    normalize=False, sublinear_tf=False, smooth_idf=True,
                    min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None):
    """
    Get a document-term matrix of shape (# docs, # unique terms) from a sequence
    of documents, each represented as a sequence of (str) terms, with a variety
    of weighting and normalization schemes for the matrix values.

    Args:
        terms_lists (Iterable[Iterable[str]]): A sequence of documents, each as
            a sequence of (str) terms. Note that the terms in each doc are to be
            counted, so these probably shouldn't be sets containing *unique*
            terms. Example inputs::

                >>> ([tok.lemma_ for tok in spacy_doc]
                ...  for spacy_doc in spacy_docs)
                >>> ((ne.text for ne in extract.named_entities(doc))
                ...  for doc in corpus)
                >>> (tuple(ng.text for ng in
                           itertools.chain.from_iterable(extract.ngrams(doc, i)
                                                         for i in range(1, 3)))
                ...  for doc in docs)

        weighting ({'tf', 'tfidf', 'binary'}): Weighting to assign to terms in
            the doc-term matrix. If 'tf', matrix values (i, j) correspond to the
            number of occurrences of term j in doc i; if 'tfidf', term frequencies
            (tf) are multiplied by their corresponding inverse document frequencies
            (idf); if 'binary', all non-zero values are set equal to 1.
        normalize (bool): If True, normalize term frequencies by the
            L2 norms of the vectors.
        binarize (bool): If True, set all term frequencies > 0 equal to 1.
        sublinear_tf (bool): If True, apply sub-linear term-frequency scaling,
            i.e. tf => 1 + log(tf).
        smooth_idf (bool): If True, add 1 to all document frequencies, equivalent
            to adding a single document to the corpus containing every unique term.
        min_df (float or int): If float, value is the fractional proportion of
            the total number of documents, which must be in [0.0, 1.0]. If int,
            value is the absolute number. Filter terms whose document frequency
            is less than ``min_df``.
        max_df (float or int): If float, value is the fractional proportion of
            the total number of documents, which must be in [0.0, 1.0]. If int,
            value is the absolute number. Filter terms whose document frequency
            is greater than ``max_df``.
        min_ic (float): Filter terms whose information content is less than
            ``min_ic``; value must be in [0.0, 1.0].
        max_n_terms (int): Only include terms whose document frequency is within
            the top ``max_n_terms``.

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
            of shape (# docs, # unique terms), where value (i, j) is the weight
            of term j in doc i
        dict: id to term mapping, where keys are unique integers as term ids and
            values are corresponding strings
    """
    stringstore = StringStore()
    data = []
    rows = []
    cols = []
    for row_idx, terms_list in enumerate(terms_lists):

        # an empty string always occupies index 0 in the stringstore, which causes
        # an empty first col in the doc-term matrix that we don't want;
        # so, we subtract 1 from the stringstore's assigned id
        bow = tuple((stringstore[term] - 1, count)
                    for term, count in collections.Counter(terms_list).items()
                    if term)

        data.extend(count for _, count in bow)
        cols.extend(term_id for term_id, _ in bow)
        rows.extend(itertools.repeat(row_idx, times=len(bow)))

    doc_term_matrix = sp.coo_matrix((data, (rows, cols)), dtype=int).tocsr()
    # ignore the 0-index empty string in stringstore, as above
    id_to_term = {term_id - 1: term for term_id, term in enumerate(stringstore)
                  if term_id != 0}

    # filter terms by document frequency or information content?
    if max_df != 1.0 or min_df != 1 or max_n_terms is not None:
        doc_term_matrix, id_to_term = filter_terms_by_df(
            doc_term_matrix, id_to_term,
            max_df=max_df, min_df=min_df, max_n_terms=max_n_terms)
    if min_ic != 0.0:
        doc_term_matrix, id_to_term = filter_terms_by_ic(
            doc_term_matrix, id_to_term,
            min_ic=min_ic, max_n_terms=max_n_terms)

    if weighting == 'binary':
        doc_term_matrix = binarize_mat(doc_term_matrix, threshold=0.0, copy=False)
    else:
        if sublinear_tf is True:
            doc_term_matrix = doc_term_matrix.astype(np.float64)
            np.log(doc_term_matrix.data, doc_term_matrix.data)
            doc_term_matrix.data += 1
        if weighting == 'tfidf':
            doc_term_matrix = apply_idf_weighting(doc_term_matrix,
                                                  smooth_idf=smooth_idf)

    if normalize is True:
        doc_term_matrix = normalize_mat(doc_term_matrix,
                                        norm='l2', axis=1, copy=False)

    return (doc_term_matrix, id_to_term)


def apply_idf_weighting(doc_term_matrix, smooth_idf=True):
    """
    Apply inverse document frequency (idf) weighting to a term-frequency (tf)
    weighted document-term matrix, optionally smoothing idf values.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms
        smooth_idf (bool): if True, add 1 to all document frequencies, equivalent
            to adding a single document to the corpus containing every unique term

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
            of shape (# docs, # unique terms), where value (i, j) is the tfidf
            weight of term j in doc i
    """
    dfs = get_doc_freqs(doc_term_matrix, normalized=False)
    n_docs = doc_term_matrix.shape[0]
    if smooth_idf is True:
        n_docs += 1
        dfs += 1
    idfs = np.log(n_docs / dfs) + 1.0
    # return doc_term_matrix.multiply(idfs)
    return doc_term_matrix.dot(sp.diags(idfs, 0))  # same as above, but 100x faster?!


def get_term_freqs(doc_term_matrix, normalized=True):
    """
    Compute absolute or relative term frequencies for all terms in a
    document-term matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms

            Note: Weighting on the terms DOES matter! Only absolute term counts
            (rather than normalized term frequencies) should be used here
        normalized (bool): if True, return normalized term frequencies, i.e.
            term counts divided by the total number of terms; if False, return
            absolute term counts

    Returns:
        :class:`numpy.ndarray <numpy.ndarray>`: array of absolute or relative term
            frequencies, with length equal to the # of unique terms, i.e. # of
            columns in ``doc_term_matrix``

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError('term-document matrix must have at least 1 non-zero entry')
    _, n_terms = doc_term_matrix.shape
    tfs = np.asarray(doc_term_matrix.sum(axis=0)).ravel()
    if normalized is True:
        return tfs / n_terms
    else:
        return tfs


def get_doc_freqs(doc_term_matrix, normalized=True):
    """
    Compute absolute or relative document frequencies for all terms in a
    term-document matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms

            Note: Weighting on the terms doesn't matter! Could be 'tf' or 'tfidf'
            or 'binary' weighting, a term's doc freq will be the same
        normalized (bool): if True, return normalized doc frequencies, i.e.
            doc counts divided by the total number of docs; if False, return
            absolute doc counts

    Returns:
        :class:`numpy.ndarray`: array of absolute or relative document
            frequencies, with length equal to the # of unique terms, i.e. # of
            columns in ``doc_term_matrix``

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError('term-document matrix must have at least 1 non-zero entry')
    n_docs, n_terms = doc_term_matrix.shape
    dfs = np.bincount(doc_term_matrix.indices, minlength=n_terms)
    if normalized is True:
        return dfs / n_docs
    else:
        return dfs


def get_information_content(doc_term_matrix):
    """
    Compute information content for all terms in a term-document matrix. IC is a
    float in [0.0, 1.0], defined as ``-df * log2(df) - (1 - df) * log2(1 - df)``,
    where df is a term's normalized document frequency.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms

            Note: Weighting on the terms doesn't matter! Could be 'tf' or 'tfidf'
            or 'binary' weighting, a term's information content will be the same

    Returns:
        :class:`numpy.ndarray`: array of term information content values,
            with length equal to the # of unique terms, i.e. # of
            columns in ``doc_term_matrix``

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    dfs = get_doc_freqs(doc_term_matrix, normalized=True)
    ics = -dfs * np.log2(dfs) - (1 - dfs) * np.log2(1 - dfs)
    ics[np.isnan(ics)] = 0.0  # NaN values not permitted!
    return ics


def filter_terms_by_df(doc_term_matrix, id_to_term,
                       max_df=1.0, min_df=1, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by document frequency),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms
        id_to_term (dict): mapping of unique integer term identifiers to their
            corresponding normalized strings
        min_df (float or int): if float, value is the fractional proportion of
            the total number of documents and must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is less than ``min_df``
        max_df (float or int): if float, value is the fractional proportion of
            the total number of documents and must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is greater than ``max_df``
        max_n_terms (int): only include terms whose *term* frequency is within
            the top `max_n_terms`

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
            of shape (# docs, # unique *filtered* terms), where value (i, j) is the
            weight of term j in doc i
        dict: id to term mapping, where keys are unique *filtered* integers as
            term ids and values are corresponding strings

    Raises:
        ValueError: if ``max_df`` or ``min_df`` or ``max_n_terms`` < 0
    """
    if max_df == 1.0 and min_df == 1 and max_n_terms is None:
        return doc_term_matrix, id_to_term
    if max_df < 0 or min_df < 0 or (max_n_terms is not None and max_n_terms < 0):
        raise ValueError('max_df, min_df, and max_n_terms may not be negative')

    n_docs, n_terms = doc_term_matrix.shape
    max_doc_count = max_df if isinstance(max_df, int) else int(max_df * n_docs)
    min_doc_count = min_df if isinstance(min_df, int) else int(min_df * n_docs)
    if max_doc_count < min_doc_count:
        raise ValueError('max_df corresponds to fewer documents than min_df')

    # calculate a mask based on document frequencies
    dfs = get_doc_freqs(doc_term_matrix, normalized=False)
    mask = np.ones(n_terms, dtype=bool)
    if max_doc_count < n_docs:
        mask &= dfs <= max_doc_count
    if min_doc_count > 1:
        mask &= dfs >= min_doc_count
    if max_n_terms is not None and mask.sum() > max_n_terms:
        tfs = get_term_freqs(doc_term_matrix, normalized=False)
        top_mask_inds = (tfs[mask]).argsort()[::-1][:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    id_to_term = {new_indices[old_index]: term
                  for old_index, term in id_to_term.items()
                  if mask[old_index]}

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        msg = 'After filtering, no terms remain; try a lower `min_df` or higher `max_df`'
        raise ValueError(msg)

    return (doc_term_matrix[:, kept_indices], id_to_term)


def filter_terms_by_ic(doc_term_matrix, id_to_term,
                       min_ic=0.0, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by information content),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms
        id_to_term (dict): mapping of unique integer term identifiers to
            corresponding normalized strings as values
        min_ic (float): filter terms whose information content is less than this
            value; must be in [0.0, 1.0]
        max_n_terms (int): only include terms whose information content is within
            the top ``max_n_terms``

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
            of shape (# docs, # unique *filtered* terms), where value (i, j) is the
            weight of term j in doc i
        dict: id to term mapping, where keys are unique *filtered* integers as
            term ids and values are corresponding strings

    Raises:
        ValueError: if ``min_ic`` not in [0.0, 1.0] or ``max_n_terms`` < 0
    """
    if min_ic == 0.0 and max_n_terms is None:
        return doc_term_matrix, id_to_term
    if min_ic < 0.0 or min_ic > 1.0:
        raise ValueError('min_ic must be a float in [0.0, 1.0]')
    if max_n_terms is not None and max_n_terms < 0:
        raise ValueError('max_n_terms may not be negative')

    _, n_terms = doc_term_matrix.shape

    # calculate a mask based on document frequencies
    ics = get_information_content(doc_term_matrix)
    mask = np.ones(n_terms, dtype=bool)
    if min_ic > 0.0:
        mask &= ics >= min_ic
    if max_n_terms is not None and mask.sum() > max_n_terms:
        top_mask_inds = (ics[mask]).argsort()[::-1][:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    id_to_term = {new_indices[old_index]: term
                  for old_index, term in id_to_term.items()
                  if mask[old_index]}

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError('After filtering, no terms remain; try a lower `min_ic`')

    return (doc_term_matrix[:, kept_indices], id_to_term)
