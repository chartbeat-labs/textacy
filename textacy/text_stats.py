"""
Statistical functions dealing with text. Kind of a grab-bag at the moment.
Includes calculations for common "readability" statistics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from math import sqrt

from textacy import data


def get_term_freqs(term_doc_matrix, normalized=True):
    """
    Compute document frequencies for all terms in a term-document matrix.

    Args:
        term_doc_matrix (:class:`scipy.sparse.csr_matrix`): N X M matrix, where
            N is the # of docs and M is the # of unique terms

            Note: Weighting on the terms DOES matter! Only absolute term counts
            (rather than normalized term frequencies) should be used here.
        normalized (bool, optional): if True, return normalized term frequencies,
            i.e. term counts divided by the total number of terms; if False,
            return absolute term counts

    Returns:
        :class:`numpy.ndarray`: length is equal to the number of unique terms,
            i.e. columns in `term_doc_matrix`

    Raises:
        ValueError: if `term_doc_matrix` doesn't have any non-zero entries
    """
    if term_doc_matrix.nnz == 0:
        raise ValueError('term-document matrix must have at least 1 non-zero entry')
    _, n_terms = term_doc_matrix.shape
    tfs = np.asarray(term_doc_matrix.sum(axis=0)).ravel()
    if normalized is True:
        return tfs / n_terms
    else:
        return tfs


def get_doc_freqs(term_doc_matrix, normalized=True):
    """
    Compute document frequencies for all terms in a term-document matrix.

    Args:
        term_doc_matrix (:class:`scipy.sparse.csr_matrix`): N X M matrix, where
            N is the # of docs and M is the # of unique terms

            Note: Weighting on the terms doesn't matter!
        normalized (bool, optional): if True, return normalized doc frequencies,
            i.e. doc counts divided by the total number of docs; if False,
            return absolute doc counts

    Returns:
        :class:`numpy.ndarray`: length is equal to the number of unique terms,
            i.e. columns in `term_doc_matrix`

    Raises:
        ValueError: if `term_doc_matrix` doesn't have any non-zero entries
    """
    if term_doc_matrix.nnz == 0:
        raise ValueError('term-document matrix must have at least 1 non-zero entry')
    n_docs, n_terms = term_doc_matrix.shape
    dfs = np.bincount(term_doc_matrix.indices, minlength=n_terms)
    if normalized is True:
        return dfs / n_docs
    else:
        return dfs


def get_information_content(term_doc_matrix):
    """
    Compute information content for all terms in a term-document matrix. IC is a
    float in [0.0, 1.0], defined as -df * log2(df) - (1 - df) * log2(1 - df),
    where df is a term's normalized document frequency.

    Args:
        term_doc_matrix (:class:`scipy.sparse.csr_matrix`): N X M matrix, where
            N is the # of docs and M is the # of unique terms

            Note: Weighting on the terms doesn't matter!

    Returns:
        :class:`numpy.ndarray`: length is equal to the number of unique terms,
            i.e. columns in `term_doc_matrix`

    Raises:
        ValueError: if `term_doc_matrix` doesn't have any non-zero entries
    """
    dfs = get_doc_freqs(term_doc_matrix, normalized=True)
    ics = -dfs * np.log2(dfs) - (1 - dfs) * np.log2(1 - dfs)
    ics[np.isnan(ics)] = 0.0  # NaN values not permitted!
    return ics


def filter_terms_by_df(term_doc_matrix, id_to_term,
                       max_df=1.0, min_df=1, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by document frequency),
    and compactify the top `max_n_terms` in the `id_to_term` mapping accordingly.
    Borrows heavily from the sklearn.feature_extraction.text module.

    Args:
        term_doc_matrix (:class:`scipy.sparse.csr_matrix`): N X M matrix, where
            N is the # of docs and M is the # of unique terms
        id_to_term (dict): mapping of unique integer term identifiers to
            their corresponding normalized strings
        min_df (float in [0.0, 1.0] or int, optional): if float, value is the
            fractional proportion of the total number of documents and must be
            in [0.0, 1.0]; if int, value is the absolute number; filter terms
            whose document frequency is less than `min_df`
        max_df (float in [0.0, 1.0] or int, optional): if float, value is the
            fractional proportion of the total number of documents and must be
            in [0.0, 1.0]; if int, value is the absolute number; filter terms
            whose document frequency is greater than `max_df`
        max_n_terms (int, optional): only include terms whose *term* frequency
            is within the top `max_n_terms`

    Returns:
        (:class:`scipy.sparse.csr_matrix`, dict): 2-tuple of the filtered
            `term_doc_matrix` and `id_to_term`

    Raises:
        ValueError: if `max_df` or `min_df` or `max_n_terms` < 0
    """
    if max_df == 1.0 and min_df == 1 and max_n_terms is None:
        return term_doc_matrix, id_to_term
    if max_df < 0 or min_df < 0 or (max_n_terms is not None and max_n_terms < 0):
        raise ValueError('max_df, min_df, and max_n_terms may not be negative')

    n_docs, n_terms = term_doc_matrix.shape
    max_doc_count = max_df if isinstance(max_df, int) else int(max_df * n_docs)
    min_doc_count = min_df if isinstance(min_df, int) else int(min_df * n_docs)
    if max_doc_count < min_doc_count:
        raise ValueError('max_df corresponds to fewer documents than min_df')

    # calculate a mask based on document frequencies
    dfs = get_doc_freqs(term_doc_matrix, normalized=False)
    mask = np.ones(n_terms, dtype=bool)
    if max_doc_count < n_docs:
        mask &= dfs <= max_doc_count
    if min_doc_count > 1:
        mask &= dfs >= min_doc_count
    if max_n_terms is not None and mask.sum() > max_n_terms:
        tfs = get_term_freqs(term_doc_matrix, normalized=False)
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

    return term_doc_matrix[:, kept_indices], id_to_term


def filter_terms_by_ic(term_doc_matrix, id_to_term,
                       min_ic=0.0, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by information content),
    and compactify the top `max_n_terms` in the `id_to_term` mapping accordingly.
    Borrows heavily from the sklearn.feature_extraction.text module.

    Args:
        term_doc_matrix (:class:`scipy.sparse.csr_matrix`)
        id_to_term (dict): mapping of unique integer term identifiers to
            corresponding normalized strings as values
        min_ic (float, optional): filter terms whose information content is less
            than this value; must be in [0.0, 1.0]
        max_n_terms (int, optional): only include terms whose information content
            is within the top `max_n_terms`

    Returns:
        (:class:`scipy.sparse.csr_matrix`, dict): 2-tuple of the filtered
            `term_doc_matrix` and `id_to_term`

    Raises:
        ValueError: if `min_ic` not in [0.0, 1.0] or `max_n_terms` < 0
    """
    if min_ic == 0.0 and max_n_terms is None:
        return term_doc_matrix, id_to_term
    if min_ic < 0.0 or min_ic > 1.0:
        raise ValueError('min_ic must be a float in [0.0, 1.0]')
    if max_n_terms is not None and max_n_terms < 0:
        raise ValueError('max_n_terms may not be negative')

    _, n_terms = term_doc_matrix.shape

    # calculate a mask based on document frequencies
    ics = get_information_content(term_doc_matrix)
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

    return term_doc_matrix[:, kept_indices], id_to_term


def readability_stats(doc):
    """
    Get calculated values for a variety of statistics related to the "readability"
    of a text: Flesch-Kincaid Grade Level, Flesch Reading Ease, SMOG Index,
    Gunning-Fog Index, Coleman-Liau Index, and Automated Readability Index.

    Also includes constituent values needed to compute the stats, e.g. word count.

    Args:
        doc (:class:`texts.TextDoc()`)

    Returns:
        dict: mapping of readability statistic name (str) to value (int or float)

    Raises:
        NotImplementedError: if `doc` is not English language. sorry.
    """
    if doc.lang != 'en':
        raise NotImplementedError('non-English NLP is not ready yet, sorry')

    n_sents = doc.n_sents

    words = doc.words(filter_punct=True)
    n_words = len(words)
    n_unique_words = len({word.lower for word in words})
    n_chars = sum(len(word) for word in words)

    hyphenator = data.load_hyphenator(lang='en')
    syllables_per_word = [len(hyphenator.positions(word.lower_)) + 1 for word in words]
    n_syllables = sum(syllables_per_word)
    n_polysyllable_words = sum(1 for n in syllables_per_word if n >= 3)

    return {'n_sents': n_sents,
            'n_words': n_words,
            'n_unique_words': n_unique_words,
            'n_chars': n_chars,
            'n_syllables': n_syllables,
            'n_polysyllable_words': n_polysyllable_words,
            'flesch_kincaid_grade_level': flesch_kincaid_grade_level(n_syllables, n_words, n_sents),
            'flesch_readability_ease': flesch_readability_ease(n_syllables, n_words, n_sents),
            'smog_index': smog_index(n_polysyllable_words, n_sents),
            'gunning_fog_index': gunning_fog_index(n_words, n_polysyllable_words, n_sents),
            'coleman_liau_index': coleman_liau_index(n_chars, n_words, n_sents),
            'automated_readability_index': automated_readability_index(n_chars, n_words, n_sents)}


def flesch_kincaid_grade_level(n_syllables, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_grade_level"""
    return 11.8 * (n_syllables / n_words) + 0.39 * (n_words / n_sents) - 15.59


def flesch_readability_ease(n_syllables, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease"""
    return -84.6 * (n_syllables / n_words) - 1.015 * (n_words / n_sents) + 206.835


def smog_index(n_polysyllable_words, n_sents, verbose=False):
    """https://en.wikipedia.org/wiki/SMOG"""
    if verbose and n_sents < 30:
        print('**WARNING: SMOG score may be unreliable for n_sents < 30')
    return 1.0430 * sqrt(30 * (n_polysyllable_words / n_sents)) + 3.1291


def gunning_fog_index(n_words, n_polysyllable_words, n_sents):
    """https://en.wikipedia.org/wiki/Gunning_fog_index"""
    return 0.4 * ((n_words / n_sents) + 100 * (n_polysyllable_words / n_words))


def coleman_liau_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index"""
    return 5.879851 * (n_chars / n_words) - 29.587280 * (n_sents / n_words) - 15.800804


def automated_readability_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Automated_readability_index"""
    return 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
