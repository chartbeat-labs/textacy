# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import math
import operator

import networkx as nx
from cytoolz import itertoolz

from . import utils as ke_utils
from .. import compat, extract, utils


def sgrank(
    doc,
    ngrams=(1, 2, 3, 4, 5, 6),
    normalize="lemma",
    window_size=1500,
    topn=10,
    idf=None,
):
    """
    Extract key terms from a document using the SGRank algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3, 4, 5, 6)``
            (default) includes all ngrams from 1 to 6; `2` if only bigrams are wanted
        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``Span`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        window_size (int): Size of sliding window in which term
            co-occurrences are determined to occur. Note: Larger values may
            dramatically increase runtime, owing to the larger number of
            co-occurrence combinations that must be counted.
        topn (int or float): Number of top-ranked terms to return as
            keyterms. If int, represents the absolute number; if float, must be
            in the open interval (0.0, 1.0), and is converted to an integer by
            ``int(round(len(candidates) * topn))``
        idf (dict): Mapping of ``normalize(term)`` to inverse document frequency
            for re-weighting of unigrams (n-grams with n > 1 have df assumed = 1).
            Results are typically better with idf information.

    Returns:
        List[Tuple[str, float]]: Sorted list of top ``topn`` key terms and
        their corresponding SGRank scores

    Raises:
        ValueError: if ``topn`` is a float but not in (0.0, 1.0] or
            ``window_size`` < 2

    Reference:
        Danesh, Sumner, and Martin. "SGRank: Combining Statistical and Graphical
        Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction."
        Lexical and Computational Semantics (* SEM 2015) (2015): 117.
    """
    ngrams = utils.to_collection(ngrams, int, tuple)
    if window_size < 2:
        raise ValueError("`window_size` must be >= 2")
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                "`topn` must be an int, or a float between 0.0 and 1.0"
            )

    n_toks = len(doc)
    window_size = min(n_toks, window_size)
    # no keyterms to extract from a (nearly) empty doc...
    if n_toks < 2:
        return []

    # if inverse doc freqs available, include nouns, adjectives, and verbs;
    # otherwise, just include nouns and adjectives
    # (without IDF downweighting, verbs dominate the results in a bad way)
    include_pos = {"NOUN", "PROPN", "ADJ", "VERB"} if idf else {"NOUN", "PROPN", "ADJ"}
    candidates, candidate_counts = _get_candidates(
        doc, ngrams, normalize, include_pos=include_pos)
    # scale float topn based on total number of initial candidates
    if isinstance(topn, float):
        topn = int(round(len(candidate_counts) * topn))
    candidates, unique_candidates = _prefilter_candidates(
        candidates, candidate_counts, topn, idf)

    term_weights = _compute_term_weights(
        candidates, candidate_counts, unique_candidates, n_toks, idf)
    # filter terms to only those with positive weights
    candidates = [cand for cand in candidates if term_weights[cand[0]] > 0]
    edge_weights = _compute_edge_weights(candidates, term_weights, window_size, n_toks)

    # build the weighted directed graph from edges, rank nodes by pagerank
    graph = nx.DiGraph()
    graph.add_edges_from(edge_weights)
    term_ranks = nx.pagerank_scipy(graph, alpha=0.85, weight="weight")
    sorted_term_ranks = sorted(
        term_ranks.items(), key=operator.itemgetter(1, 0), reverse=True)

    return ke_utils.get_filtered_topn_terms(
        sorted_term_ranks, topn, match_threshold=0.8)


def _get_candidates(
    doc, ngrams, normalize, include_pos=("NOUN", "PROPN", "ADJ", "VERB")
):
    """
    Get n-gram candidate keyterms from ``doc``, with key information for each:
    its normalized text string, position within the doc, number of constituent words,
    and frequency of occurrence.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (Tuple[int])
        normalize (str)
        include_pos (Set[str])

    Returns:
        List[Tuple[str, int, int, int]]
        Dict[str, int]
    """
    min_term_freq = min(max(len(doc) // 1000, 1), 4)
    candidates_tuples = list(
        ke_utils.get_ngram_candidates(doc, ngrams, include_pos=include_pos))
    candidate_texts = [
        " ".join(ke_utils.normalize_terms(ctup, normalize))
        for ctup in candidates_tuples
    ]
    candidate_counts = collections.Counter(candidate_texts)
    if min_term_freq > 1:
        candidates = [
            (ctext, ctup[0].i, len(ctup), candidate_counts[ctext])
            for ctup, ctext in compat.zip_(candidates_tuples, candidate_texts)
            if candidate_counts[ctext] >= min_term_freq
        ]
    else:
        candidates = [
            (ctext, ctup[0].i, len(ctup), candidate_counts[ctext])
            for ctup, ctext in compat.zip_(candidates_tuples, candidate_texts)
        ]
    return candidates, candidate_counts


def _prefilter_candidates(candidates, candidate_counts, topn, idf):
    """
    Filter initial set of candidates to only those with sufficiently high TF or
    (if available) modified TF*IDF.

    Args:
        candidates (List[Tuple[str, int, int, int]])
        candidate_counts (Dict[str, int])
        topn (int)
        idf (Dict[str, float])

    Returns:
        List[Tuple[str, int, int, int]]
        Set[str]
    """
    topn_prefilter = max(3 * topn, 100)
    if idf:
        mod_tfidfs = {
            ctext: ccount * idf.get(ctext, 1) if " " not in ctext else ccount
            for ctext, _, _, ccount in candidates
        }
        unique_candidates = {
            ctext
            for ctext, _ in sorted(
                mod_tfidfs.items(), key=operator.itemgetter(1), reverse=True
            )[:topn_prefilter]
        }
    else:
        unique_candidates = {
            ctext for ctext, _ in candidate_counts.most_common(topn_prefilter)}
    candidates = [cand for cand in candidates if cand[0] in unique_candidates]
    return candidates, unique_candidates


def _compute_term_weights(candidates, candidate_counts, unique_candidates, n_toks, idf):
    """
    Compute term weights from statistical attributes: position of first occurrence,
    not subsumed frequency, and number of constituent words.

    Args:
        candidates (List[Tuple[str, int, int, int]])
        candidate_counts (Dict[str, int])
        unique_candidates (Set[str])
        n_toks (int)
        idf (Dict[str, float])

    Returns:
        Dict[str, float]
    """
    term_weights = {}
    seen_terms = set()
    n_toks_p1 = n_toks + 1
    for ctext, cpos, clen, ccount in candidates:
        # we only want the *first* occurrence of a unique term (by its text)
        if ctext in seen_terms:
            continue
        else:
            seen_terms.add(ctext)
        pos_first_occ = math.log(n_toks_p1 / (cpos + 1))
        # TODO: do we want to sub-linear scale term len or not?
        clen = math.sqrt(clen)
        # subtract from subsum_count for the case when ctext == ctext2
        subsum_count = sum(
            candidate_counts[ctext2]
            for ctext2 in unique_candidates
            if ctext in ctext2
        ) - ccount
        term_freq_factor = ccount - subsum_count
        if idf and clen == 1:
            term_freq_factor *= idf.get(ctext, 1)
        term_weights[ctext] = term_freq_factor * pos_first_occ * clen
    return term_weights


def _compute_edge_weights(candidates, term_weights, window_size, n_toks):
    """
    Compute weights between candidates that occur within a sliding window(s) of
    each other, then combine with statistical ``term_weights`` and normalize
    by the total number of outgoing edge weights.

    Args:
        candidates (List[Tuple[str, int, int, int]])
        term_weights (Dict[str, float])
        window_size (int)
        n_toks (int)

    Returns:
        Dict[str, float]
    """
    n_coocs = collections.defaultdict(lambda: collections.defaultdict(int))
    sum_logdists = collections.defaultdict(lambda: collections.defaultdict(float))
    # iterate over windows
    log_ = math.log  # localize this, for performance
    for start_ind in compat.range_(n_toks):
        end_ind = start_ind + window_size
        window_cands = (cand for cand in candidates if start_ind <= cand[1] < end_ind)
        # get all token combinations within window
        for c1, c2 in itertools.combinations(window_cands, 2):
            n_coocs[c1[0]][c2[0]] += 1
            try:
                sum_logdists[c1[0]][c2[0]] += log_(window_size / abs(c1[1] - c2[1]))
            except ZeroDivisionError:
                sum_logdists[c1[0]][c2[0]] += log_(window_size)
        if end_ind >= n_toks:
            break
    # compute edge weights between co-occurring terms (nodes)
    edge_weights = collections.defaultdict(lambda: collections.defaultdict(float))
    for c1, c2s in sum_logdists.items():
        for c2 in c2s:
            edge_weights[c1][c2] = (
                ((1.0 + sum_logdists[c1][c2]) / n_coocs[c1][c2])
                * term_weights[c1]
                * term_weights[c2]
            )
    # normalize edge weights by sum of outgoing edge weights per term (node)
    norm_edge_weights = []
    for c1, c2s in edge_weights.items():
        sum_edge_weights = sum(c2s.values())
        norm_edge_weights.extend(
            (c1, c2, {"weight": weight / sum_edge_weights})
            for c2, weight in c2s.items()
        )
    return norm_edge_weights
