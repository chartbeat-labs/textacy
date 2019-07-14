# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import operator

import networkx as nx
from cytoolz import itertoolz

from . import utils
from .. import compat


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

    # bail out on empty docs
    if not doc:
        return []

    # build up a graph of good words, edges weighting by adjacent sentence co-occurrence
    cooc_mat = collections.Counter()
    n_sents = itertoolz.count(doc.sents)  # in case doc only has 1 sentence
    for sent1, sent2 in itertoolz.sliding_window(min(2, n_sents), doc.sents):
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
    # doc doesn't have any valid words...
    if not cooc_mat:
        return []

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
    word_strs = list(graph.nodes())
    # "level of hierarchy" component
    max_truss_levels = _compute_node_truss_levels(graph)
    max_truss_level = max(max_truss_levels.values())
    # "semantic strength of a word" component
    sem_strengths = {
        w: sum(cooc_mat[tuple(sorted([w, nbr]))] * max_truss_levels[nbr] for nbr in graph.neighbors(w))
        for w in word_strs
    }
    # "semantic connectivity" component
    sem_connectivities = {
        w: len(set(max_truss_levels[nbr] for nbr in graph.neighbors(w))) / max_truss_level
        for w in word_strs
    }
    # "positional weight" component
    word_pos = collections.defaultdict(float)
    for word, word_str in compat.zip_(doc, utils.normalize_terms(doc, normalize)):
        word_pos[word_str] += 1 / (word.i + 1)
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


def _compute_node_truss_levels(graph):
    """
    Args:
        graph (:class:`networkx.Graph`)

    Returns:
        Dict[str, int]

    Reference:
        Burkhardt, Paul & Faber, Vance & G. Harris, David. (2018).
        Bounds and algorithms for $k$-truss.
        https://arxiv.org/abs/1806.05523v1
    """
    max_edge_ks = {}
    is_removed = collections.defaultdict(int)
    triangle_counts = {
        edge: len(set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1])))
        for edge in graph.edges()}
    # rather than iterating over all theoretical values of k
    # let's break out early once all edges have been removed
    # max_edge_k = math.ceil(math.sqrt(len(triangle_counts)))
    # for k in range(1, max_edge_k):
    k = 1
    while True:
        to_remove = collections.deque(
            edge
            for edge, tcount in triangle_counts.items()
            if tcount < k and not is_removed[edge]
        )
        while to_remove:
            edge = to_remove.popleft()
            is_removed[edge] = 1
            for nbr in (set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1]))):
                for node in edge:
                    nbr_edge = (node, nbr)
                    try:
                        triangle_counts[nbr_edge] -= 1
                    except KeyError:
                        # oops, gotta reverse the node ordering on this edge
                        nbr_edge = (nbr, node)
                        triangle_counts[nbr_edge] -= 1
                    if triangle_counts[nbr_edge] == k - 1:
                        to_remove.append(nbr_edge)
                        is_removed[nbr_edge] = 1
            max_edge_ks[edge] = k - 1
        # here's where we break out early, if possible
        if len(is_removed) == len(triangle_counts):
            break
        else:
            k += 1
    max_node_ks = {
        node: max(k for edge, k in max_edge_ks.items() if node in edge)
        for node in graph.nodes()
    }
    return max_node_ks
