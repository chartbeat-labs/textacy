# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import operator

import networkx as nx
from cytoolz import itertoolz

from . import utils
from .. import compat, utils


def scake(
    doc,
    normalize="lemma",
    include_pos=("NOUN", "PROPN", "ADJ"),
    topn=10,
):
    """
    Extract key terms from a document using the sCAKE algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`): A spaCy ``Doc``. Must be sentence-segmented
            and (optionally) POS-tagged.
        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        include_pos (Set[str])
        topn (int or float): Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(candidates) * topn))``

    Returns:
        List[Tuple[str, float]]: Sorted list of top ``topn`` key terms and
        their corresponding scores.

    Reference:
        Duari, Swagata & Bhatnagar, Vasudha. (2018). sCAKE: Semantic Connectivity
        Aware Keyword Extraction. Information Sciences. 477.
        https://arxiv.org/abs/1811.10831v1
    """
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                "topn={} is invalid; "
                "must be an int, or a float between 0.0 and 1.0".format(topn)
            )

    # build up a graph of good words, edges weighting by adjacent sentence co-occurrence
    cooc_mat = collections.Counter()
    for sent1, sent2 in itertoolz.sliding_window(2, doc.sents):
        window_words = (
            word
            for word in itertoolz.concatv(sent1, sent2)
            if not (word.is_stop or word.is_punct or word.is_space)
            and (not include_pos or word.pos_ in include_pos)
        )
        window_words = utils.normalize_terms(window_words, normalize)
        cooc_mat.update(
            w1_w2
            for w1_w2 in itertools.combinations(sorted(window_words), 2)
            if w1_w2[0] != w1_w2[1]
        )
    graph = nx.Graph()
    graph.add_edges_from(
        (w1, w2, {"weight": weight})
        for (w1, w2), weight in cooc_mat.items()
    )

    word_scores = _compute_word_scores(doc, graph, cooc_mat, normalize)

    # generate a list of candidate terms
    candidates = _get_candidates(doc, include_pos, normalize)
    if isinstance(topn, float):
        topn = int(round(len(set(candidates)) * topn))
    # rank candidates by aggregating constituent word scores
    candidate_scores = {
        " ".join(candidate): sum(word_scores.get(word, 0.0) for word in candidate)
        for candidate in candidates
    }
    sorted_candidate_scores = sorted(
        candidate_scores.items(), key=operator.itemgetter(1, 0), reverse=True)
    return utils.get_filtered_topn_terms(
        sorted_candidate_scores, topn, match_threshold=0.8)


def _compute_word_scores(doc, graph, cooc_mat, normalize):
    """
    Args:
        doc (:class:`spacy.tokens.Doc`)
        graph (:class:`networkx.Graph`)
        cooc_mat (Dict[Tuple[str, str], int])
        normalize (str)

    Returns:
        Dict[str, float]
    """
    word_pos = collections.defaultdict(float)
    all_word_strs = utils.normalize_terms(doc, normalize)
    for word, word_str in compat.zip_(doc, all_word_strs):
        word_pos[word_str] += 1 / (word.i + 1)

    word_strs = list(graph.nodes)
    truss_levels = collections.defaultdict(list)
    i = 1
    while True:
        subgraph = _get_ktruss(graph, i)
        if len(subgraph) == 0:
            break
        for word in subgraph.nodes:
            truss_levels[word].append(i)
        i += 1

    max_truss_level = i - 1
    max_truss_levels = {w: max(truss_levels[w]) for w in word_strs}
    sem_strengths = {
        w: sum(cooc_mat[tuple(sorted([w, nbr]))] * max_truss_levels[nbr] for nbr in graph[w])
        for w in word_strs
    }
    sem_connectivities = {
        w: len(set(itertoolz.concat(truss_levels[nbr] for nbr in graph[w]))) / max_truss_level
        for w in word_strs
    }
    return {
        w: word_pos[w] * max_truss_levels[w] * sem_strengths[w] * sem_connectivities[w]
        for w in word_strs
    }


def _get_candidates(doc, include_pos, normalize):
    """
    Get a set of candidate terms to be scored by joining the longest
    subsequences of valid words -- non-stopword and non-punct, filtered to
    nouns, proper nouns, and adjectives if ``doc`` is POS-tagged -- then
    normalized into strings.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        include_pos (Set[str])
        normalize (str or callable)

    Returns:
        Set[Tuple[str]]
    """
    if include_pos:
        include_pos = set(include_pos)

    def _is_valid_tok(tok):
        return (
            not (tok.is_stop or tok.is_punct or tok.is_space)
            and (not include_pos or tok.pos_ in include_pos)
        )

    candidates = utils.get_consecutive_subsequences(doc, _is_valid_tok)
    return {
        tuple(utils.normalize_terms(candidate, normalize))
        for candidate in candidates
    }


def _get_ktruss(G, k):
    """
    Args:
        G (:class:`networkx.Graph`)
        k (int)

    Returns:
        :class:`networkx.Graph`
    """
    H = G.copy()
    n_dropped = 1
    while n_dropped > 0:
        n_dropped = 0
        to_drop = []
        seen = set()
        for u in H:
            seen.add(u)
            nbrs = set(H[u])
            to_drop.extend(
                (u, v)
                for v in nbrs
                if v not in seen
                and len(nbrs & set(H[v])) < k
            )
        H.remove_edges_from(to_drop)
        H.remove_nodes_from(list(nx.isolates(H)))
        n_dropped = len(to_drop)
    return H
