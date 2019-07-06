# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import operator

from . import graph_base, utils
from .. import compat


def textrank(doc, normalize="lemma", window_size=2, topn=10):
    """
    Extract key terms from a document using the TextRank algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        window_size (int): Size of sliding window in which term co-occurrences
            are determined.
        topn (int or float): Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(set(candidates)) * topn))``.

    Returns:
        List[Tuple[str, float]]: Sorted list of top ``topn`` key terms and
        their corresponding TextRank ranking scores.

    References:
        Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts.
        Association for Computational Linguistics.
    """
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                "topn={} is invalid; "
                "must be an int, or a float between 0.0 and 1.0".format(topn)
            )

    # build graph from all words in doc
    graph = graph_base.build_graph_from_terms(
        [word for word in doc],
        normalize=normalize,
        window_size=window_size,
        edge_weighting="binary",
    )
    # generate a list of candidate terms
    candidates = _get_candidates(doc, normalize)
    if isinstance(topn, float):
        topn = int(round(len(set(candidates)) * topn))
    # rank individual words
    # then rank candidates by aggregating constituent word scores
    word_scores = graph_base.rank_nodes_by_pagerank(graph, weight=None)
    # TODO: PY3 doesn't need to make a list when computing the mean
    candidate_scores = {
        " ".join(candidate): compat.mean_([word_scores.get(word, 0.0) for word in candidate])
        for candidate in candidates
    }
    return utils.get_filtered_topn_terms(
        sorted(candidate_scores.items(), key=operator.itemgetter(1), reverse=True),
        topn,
        match_threshold=None,
    )


def _get_candidates(doc, normalize):
    """
    Get a set of candidate terms to be scored by joining the longest
    subsequences of valid words -- non-stopword and non-punct, filtered to
    nouns, proper nouns, and adjectives if ``doc`` is POS-tagged -- then
    normalized into strings.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str or callable)

    Returns:
        Set[Tuple[str]]
    """
    if doc.is_tagged:
        include_pos = {"NOUN", "PROPN", "ADJ"}
    else:
        include_pos = None

    def _is_valid_tok(tok):
        return (
            not (tok.is_stop or tok.is_punct or tok.is_space)
            and (include_pos is None or tok.pos_ in include_pos)
        )

    candidates = utils.get_consecutive_subsequences(doc, _is_valid_tok)
    return {
        tuple(utils.normalize_terms(candidate, normalize))
        for candidate in candidates
    }
