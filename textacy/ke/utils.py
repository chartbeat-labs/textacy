# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

from cytoolz import itertoolz

from .. import similarity


def normalize_terms(terms, normalize):
    """
    Transform a sequence of terms from spaCy ``Token`` or ``Span`` s into
    strings, normalized by ``normalize``.

    Args:
        terms (Sequence[:class:`spacy.tokens.Token` or :class:`spacy.tokens.Span`])
        normalize (str or Callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if falsy, use the form of terms as they appear
            in ``terms``; if a callable, must accept a ``Token`` or ``Span``
            and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`.

    Yields:
        str
    """
    if normalize == "lemma":
        terms = (term.lemma_ for term in terms)
    elif normalize == "lower":
        terms = (term.lower_ for term in terms)
    elif normalize is None:
        terms = (term.text for term in terms)
    elif callable(normalize):
        terms = (normalize(term) for term in terms)
    else:
        raise ValueError(
            "normalize = {} is invalid; value must be a function or one of {}".format(
                normalize, {"lemma", "lower", None})
        )
    for term in terms:
        yield term


def get_consecutive_subsequences(terms, grp_func):
    """
    Get consecutive subsequences of terms for which all ``grp_func(term)`` is True;
    discard terms for which the output is False.

    Args:
        terms (Sequence[object]): Ordered sequence of terms, probably as
            strings or spaCy ``Tokens`` or ``Spans``.
        grp_func (callable): Function applied sequentially to each term in ``terms``
            that returns a boolean.

    Yields:
        Tuple[object]: Next consecutive subsequence of terms in ``terms``
        for which all ``grp_func(term)`` is True, grouped together in a tuple.
    """
    for key, terms_grp in itertools.groupby(terms, key=grp_func):
        if key:
            yield tuple(terms_grp)


def get_ngram_candidates(doc, ns, include_pos=("NOUN", "PROPN", "ADJ")):
    """
    Get a sequence of good ngrams (for each n in ``ns``) to use as candidates
    in keyterm extraction algorithms, where "good" means that they don't start
    or end with a stop word or contain any punctuation-only tokens. Optionally,
    require all constituent words to have POS tags in ``include_pos``.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ns (Tuple[int])
        include_pos (Set[str])

    Yields:
        Tuple[:class:`spacy.tokens.Token`]: Next good ngram, as a tuple of Tokens.
    """
    ngrams = itertoolz.concat(itertoolz.sliding_window(n, doc) for n in ns)
    ngrams = (
        ngram
        for ngram in ngrams
        if not (ngram[0].is_stop or ngram[-1].is_stop)
        and not any(word.is_punct or word.is_space for word in ngram)
    )
    if include_pos:
        include_pos = set(include_pos)
        ngrams = (
            ngram
            for ngram in ngrams
            if all(word.pos_ in include_pos for word in ngram)
        )
    for ngram in ngrams:
        yield ngram


def get_filtered_topn_terms(term_scores, topn, match_threshold=None):
    """
    Build up a list of the ``topn`` terms, filtering out any that are substrings
    of better-scoring terms and optionally filtering out any that are sufficiently
    similar to better-scoring terms.

    Args:
        term_scores (List[Tuple[str, float]]): List of (term, score) pairs,
            sorted in order from best score to worst. Note that this may be
            from high to low value or low to high, depending on the algorithm.
        topn (int): Maximum number of top-scoring terms to get.
        match_threshold (float): Minimal edit distance between a term and previously
            seen terms, used to filter out terms that are sufficiently similar
            to higher-scoring terms. Uses :func:`textacy.similarity.token_sort_ratio()`.

    Returns:
        List[Tuple[str, float]]
    """
    topn_terms = []
    seen_terms = set()
    sim_func = similarity.token_sort_ratio
    for term, score in term_scores:
        # skip terms that are substrings of any higher-scoring term
        if any(term in st for st in seen_terms):
            continue
        # skip terms that are sufficiently similar to any higher-scoring term
        if (
            match_threshold
            and any(sim_func(term, st) >= match_threshold for st in seen_terms)
        ):
            continue
        seen_terms.add(term)
        topn_terms.append((term, score))
        if len(topn_terms) >= topn:
            break
    return topn_terms
