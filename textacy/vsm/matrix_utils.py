"""
Sparse Matrix Utils
-------------------

Functions for computing corpus-wide term- or document-based values, like
term frequency, document frequency, and document length, and filtering terms
from a matrix by their document frequency.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.sparse as sp


def get_term_freqs(doc_term_matrix, type_="linear"):
    """
    Compute frequencies for all terms in a document-term matrix, with optional
    sub-linear scaling.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs and N is the # of unique terms. Values must be
            the linear, un-scaled counts of term n per doc m.
        type_ ({'linear', 'sqrt', 'log'}): Scaling applied to absolute term counts.
            If 'linear', term counts are left as-is, since the sums are already
            linear; if 'sqrt', tf => sqrt(tf); if 'log', tf => log(tf) + 1.

    Returns:
        :class:`numpy.ndarray`: Array of term frequencies, with length equal to
        the # of unique terms (# of columns) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries, or
            if ``type_`` isn't one of {"linear", "sqrt", "log"}.
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError("`doc_term_matrix` must have at least 1 non-zero entry")
    tfs = np.asarray(doc_term_matrix.sum(axis=0)).ravel()
    if type_ == "linear":
        return tfs  # tfs is already linear
    elif type_ == "sqrt":
        return np.sqrt(tfs)
    elif type_ == "log":
        return np.log(tfs) + 1.0
    else:
        raise ValueError(
            "type_ = {} is invalid; value must be one of {}".format(
                type_, {"linear", "sqrt", "log"}
            )
        )


def get_doc_freqs(doc_term_matrix):
    """
    Compute document frequencies for all terms in a document-term matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs and N is the # of unique terms.

            .. note:: Weighting on the terms doesn't matter! Could be binary or
               tf or tfidf, a term's doc freq will be the same.

    Returns:
        :class:`numpy.ndarray`: Array of document frequencies, with length equal to
        the # of unique terms (# of columns) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries.
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError("`doc_term_matrix` must have at least 1 non-zero entry")
    _, n_terms = doc_term_matrix.shape
    return np.bincount(doc_term_matrix.indices, minlength=n_terms)


def get_inverse_doc_freqs(doc_term_matrix, type_="smooth"):
    """
    Compute inverse document frequencies for all terms in a document-term matrix,
    using one of several IDF formulations.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs and N is the # of unique terms.
            The particular weighting of matrix values doesn't matter.
        type_ ({'standard', 'smooth', 'bm25'}): Type of IDF formulation to use.
            If 'standard', idfs => log(n_docs / dfs) + 1.0;
            if 'smooth', idfs => log(n_docs + 1 / dfs + 1) + 1.0, i.e. 1 is added
            to all document frequencies, equivalent to adding a single document
            to the corpus containing every unique term;
            if 'bm25', idfs => log((n_docs - dfs + 0.5) / (dfs + 0.5)), which is
            a form commonly used in BM25 ranking that allows for extremely common
            terms to have negative idf weights.

    Returns:
        :class:`numpy.ndarray`: Array of inverse document frequencies, with length
        equal to the # of unique terms (# of columns) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``type_`` isn't one of {"standard", "smooth", "bm25"}.
    """
    dfs = get_doc_freqs(doc_term_matrix)
    n_docs, _ = doc_term_matrix.shape
    if type_ == "standard":
        return np.log(n_docs / dfs) + 1.0
    elif type_ == "smooth":
        n_docs += 1
        dfs += 1
        return np.log(n_docs / dfs) + 1.0
    elif type_ == "bm25":
        return np.log((n_docs - dfs + 0.5) / (dfs + 0.5))
    else:
        raise ValueError(
            "type_ = {} is invalid; value must be one of {}".format(
                type_, {"standard", "smooth", "bm25"}
            )
        )


def get_doc_lengths(doc_term_matrix, type_="linear"):
    """
    Compute the lengths (i.e. number of terms) for all documents in a
    document-term matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs, N is the # of unique terms, and values are
            the absolute counts of term n per doc m.
        type_ ({'linear', 'sqrt', 'log'}): Scaling applied to absolute doc lengths.
            If 'linear', lengths are left as-is, since the sums are already
            linear; if 'sqrt', dl => sqrt(dl); if 'log', dl => log(dl) + 1.

    Returns:
        :class:`numpy.ndarray`: Array of document lengths, with length equal to
        the # of documents (# of rows) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``type_`` isn't one of {"linear", "sqrt", "log"}.
    """
    dls = np.asarray(doc_term_matrix.sum(axis=1)).ravel()
    if type_ == "linear":
        return dls  # dls is already linear
    elif type_ == "sqrt":
        return np.sqrt(dls)
    elif type_ == "log":
        return np.log(dls) + 1.0
    else:
        raise ValueError(
            "`type_` = {} invalid; must be one of {}".format(
                type_, {"linear", "sqrt", "log"}
            )
        )


def get_information_content(doc_term_matrix):
    """
    Compute information content for all terms in a document-term matrix. IC is a
    float in [0.0, 1.0], defined as ``-df * log2(df) - (1 - df) * log2(1 - df)``,
    where df is a term's normalized document frequency.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs and N is the # of unique terms.

            .. note:: Weighting on the terms doesn't matter! Could be binary or
               tf or tfidf, a term's information content will be the same.

    Returns:
        :class:`numpy.ndarray`: Array of term information content values, with
        length equal to the # of unique terms (# of columns) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries.
    """
    dfs = get_doc_freqs(doc_term_matrix)
    # normalize doc freqs by total number of docs
    # TODO: is this *really* what we want to do?
    dfs = dfs / doc_term_matrix.shape[0]
    ics = -dfs * np.log2(dfs) - (1 - dfs) * np.log2(1 - dfs)
    ics[np.isnan(ics)] = 0.0  # NaN values not permitted!
    return ics


def apply_idf_weighting(doc_term_matrix, type_="smooth"):
    """
    Apply inverse document frequency (idf) weighting to a term-frequency (tf)
    weighted document-term matrix, using one of several IDF formulations.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M x N sparse matrix,
            where M is the # of docs and N is the # of unique terms.
        type_ ({'standard', 'smooth', 'bm25'}): Type of IDF formulation to use.

    Returns:
        :class:`scipy.sparse.csr_matrix`: Sparse matrix of shape M x N,
        where value (i, j) is the tfidf weight of term j in doc i.

    See Also:
        :func:`get_inverse_doc_freqs()`
    """
    idfs = get_inverse_doc_freqs(doc_term_matrix, type_=type_)
    return doc_term_matrix.dot(sp.diags(idfs, 0))


def filter_terms_by_df(
    doc_term_matrix, term_to_id, max_df=1.0, min_df=1, max_n_terms=None
):
    """
    Filter out terms that are too common and/or too rare (by document frequency),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N matrix, where
            M is the # of docs and N is the # of unique terms.
        term_to_id (Dict[str, int]): Mapping of term string to unique term id,
            e.g. :attr:`Vectorizer.vocabulary_terms`.
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
        :class:`scipy.sparse.csr_matrix`: Sparse matrix of shape (# docs, # unique filtered terms),
        where value (i, j) is the weight of term j in doc i.

        Dict[str, int]: Term to id mapping, where keys are unique *filtered* terms
        as strings and values are their corresponding integer ids.

    Raises:
        ValueError: if ``max_df`` or ``min_df`` or ``max_n_terms`` < 0.
    """
    if max_df == 1.0 and min_df == 1 and max_n_terms is None:
        return doc_term_matrix, term_to_id
    if max_df < 0 or min_df < 0 or (max_n_terms is not None and max_n_terms < 0):
        raise ValueError("max_df, min_df, and max_n_terms may not be negative")

    n_docs, n_terms = doc_term_matrix.shape
    max_doc_count = max_df if isinstance(max_df, int) else int(max_df * n_docs)
    min_doc_count = min_df if isinstance(min_df, int) else int(min_df * n_docs)
    if max_doc_count < min_doc_count:
        raise ValueError("max_df corresponds to fewer documents than min_df")

    # calculate a mask based on document frequencies
    dfs = get_doc_freqs(doc_term_matrix)
    mask = np.ones(n_terms, dtype=bool)
    if max_doc_count < n_docs:
        mask &= dfs <= max_doc_count
    if min_doc_count > 1:
        mask &= dfs >= min_doc_count
    if max_n_terms is not None and mask.sum() > max_n_terms:
        tfs = get_term_freqs(doc_term_matrix, type_="linear")
        top_mask_inds = (-tfs[mask]).argsort()[:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    term_to_id = {
        term: new_indices[old_index]
        for term, old_index in term_to_id.items()
        if mask[old_index]
    }

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError(
            "After filtering, no terms remain; "
            "try a lower `min_df` or higher `max_df`"
        )

    return (doc_term_matrix[:, kept_indices], term_to_id)


def filter_terms_by_ic(doc_term_matrix, term_to_id, min_ic=0.0, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by information content),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N sparse matrix,
            where M is the # of docs and N is the # of unique terms.
        term_to_id (Dict[str, int]): Mapping of term string to unique term id,
            e.g. :attr:`Vectorizer.vocabulary_terms`.
        min_ic (float): filter terms whose information content is less than this
            value; must be in [0.0, 1.0]
        max_n_terms (int): only include terms whose information content is within
            the top ``max_n_terms``

    Returns:
        :class:`scipy.sparse.csr_matrix`: Sparse matrix of shape (# docs, # unique filtered terms),
        where value (i, j) is the weight of term j in doc i.

        Dict[str, int]: Term to id mapping, where keys are unique *filtered* terms
        as strings and values are their corresponding integer ids.

    Raises:
        ValueError: if ``min_ic`` not in [0.0, 1.0] or ``max_n_terms`` < 0.
    """
    if min_ic == 0.0 and max_n_terms is None:
        return doc_term_matrix, term_to_id
    if min_ic < 0.0 or min_ic > 1.0:
        raise ValueError("min_ic must be a float in [0.0, 1.0]")
    if max_n_terms is not None and max_n_terms < 0:
        raise ValueError("max_n_terms may not be negative")

    _, n_terms = doc_term_matrix.shape

    # calculate a mask based on document frequencies
    ics = get_information_content(doc_term_matrix)
    mask = np.ones(n_terms, dtype=bool)
    if min_ic > 0.0:
        mask &= ics >= min_ic
    if max_n_terms is not None and mask.sum() > max_n_terms:
        top_mask_inds = (-ics[mask]).argsort()[:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    term_to_id = {
        term: new_indices[old_index]
        for term, old_index in term_to_id.items()
        if mask[old_index]
    }

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError("After filtering, no terms remain; try a lower `min_ic`")

    return (doc_term_matrix[:, kept_indices], term_to_id)
