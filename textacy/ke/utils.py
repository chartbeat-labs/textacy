# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools


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
