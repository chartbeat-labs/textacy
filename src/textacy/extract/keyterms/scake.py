from __future__ import annotations

import collections
import itertools
from operator import itemgetter
from typing import Callable, Collection, Counter, Iterable, Optional

import networkx as nx
from cytoolz import itertoolz
from spacy.tokens import Doc, Token

from ... import utils
from .. import utils as ext_utils


def scake(
    doc: Doc,
    *,
    normalize: Optional[str | Callable[[Token], str]] = "lemma",
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """
    Extract key terms from a document using the sCAKE algorithm.

    Args:
        doc: spaCy ``Doc`` from which to extract keyterms. Must be sentence-segmented;
            optionally POS-tagged.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms; if None,
            use the form of terms as they appeared in ``doc``; if a callable,
            must accept a ``Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        include_pos: One or more POS tags with which to filter for good candidate keyterms.
            If None, include tokens of all POS tags
            (which also allows keyterm extraction from docs without POS-tagging.)
        topn: Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(candidates) * topn))``

    Returns:
        Sorted list of top ``topn`` key terms and their corresponding scores.

    References:
        Duari, Swagata & Bhatnagar, Vasudha. (2018). sCAKE: Semantic Connectivity
        Aware Keyword Extraction. Information Sciences. 477.
        https://arxiv.org/abs/1811.10831v1
    """
    # validate / transform args
    include_pos: Optional[set[str]] = utils.to_set(include_pos) if include_pos else None
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
    cooc_mat: Counter[tuple[str, str]] = collections.Counter()
    # handle edge case where doc only has 1 sentence
    n_sents = itertoolz.count(doc.sents)
    for window_sents in itertoolz.sliding_window(min(2, n_sents), doc.sents):
        if n_sents == 1:
            window_sents = (window_sents[0], [])
        window_words: Iterable[str] = (
            word
            for word in itertoolz.concat(window_sents)
            if not (word.is_stop or word.is_punct or word.is_space)
            and (not include_pos or word.pos_ in include_pos)
        )
        window_words = ext_utils.terms_to_strings(window_words, normalize)
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
        (w1, w2, {"weight": weight}) for (w1, w2), weight in cooc_mat.items()
    )

    word_scores = _compute_word_scores(doc, graph, cooc_mat, normalize)
    if not word_scores:
        return []

    # generate a list of candidate terms
    candidates = _get_candidates(doc, normalize, include_pos)
    if isinstance(topn, float):
        topn = int(round(len(set(candidates)) * topn))
    # rank candidates by aggregating constituent word scores
    candidate_scores = {
        " ".join(candidate): sum(word_scores.get(word, 0.0) for word in candidate)
        for candidate in candidates
    }
    sorted_candidate_scores = sorted(
        candidate_scores.items(), key=itemgetter(1, 0), reverse=True
    )
    return ext_utils.get_filtered_topn_terms(
        sorted_candidate_scores, topn, match_threshold=0.8
    )


def _compute_word_scores(
    doc: Doc,
    graph: nx.Graph,
    cooc_mat: dict[tuple[str, str], int],
    normalize: Optional[str | Callable[[Token], str]],
) -> dict[str, float]:
    word_strs: list[str] = list(graph.nodes())
    # "level of hierarchy" component
    max_truss_levels = _compute_node_truss_levels(graph)
    max_truss_level = max(max_truss_levels.values())
    # check for edge case when all word scores would be zero / undefined
    if not max_truss_level:
        return {}

    # "semantic strength of a word" component
    sem_strengths = {
        w: sum(
            cooc_mat[tuple(sorted([w, nbr]))] * max_truss_levels[nbr]
            for nbr in graph.neighbors(w)
        )
        for w in word_strs
    }
    # "semantic connectivity" component
    sem_connectivities = {
        w: len(set(max_truss_levels[nbr] for nbr in graph.neighbors(w)))
        / max_truss_level
        for w in word_strs
    }
    # "positional weight" component
    word_pos = collections.defaultdict(float)
    for word, word_str in zip(doc, ext_utils.terms_to_strings(doc, normalize)):
        word_pos[word_str] += 1 / (word.i + 1)
    return {
        w: word_pos[w] * max_truss_levels[w] * sem_strengths[w] * sem_connectivities[w]
        for w in word_strs
    }


def _get_candidates(
    doc: Doc,
    normalize: Optional[str | Callable[[Token], str]],
    include_pos: Optional[set[str]],
) -> set[tuple[str, ...]]:
    """
    Get a set of candidate terms to be scored by joining the longest
    subsequences of valid words -- non-stopword and non-punct, filtered to
    nouns, proper nouns, and adjectives if ``doc`` is POS-tagged -- then
    normalized into strings.
    """

    def _is_valid_tok(tok):
        return not (tok.is_stop or tok.is_punct or tok.is_space) and (
            not include_pos or tok.pos_ in include_pos
        )

    candidates = ext_utils.get_longest_subsequence_candidates(doc, _is_valid_tok)
    return {
        tuple(ext_utils.terms_to_strings(candidate, normalize))
        for candidate in candidates
    }


def _compute_node_truss_levels(graph: nx.Graph) -> dict[str, int]:
    """
    Reference:
        Burkhardt, Paul & Faber, Vance & G. Harris, David. (2018).
        Bounds and algorithms for $k$-truss.
        https://arxiv.org/abs/1806.05523v1
    """
    max_edge_ks = {}
    is_removed = collections.defaultdict(int)
    triangle_counts = {
        edge: len(set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1])))
        for edge in graph.edges()
    }
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
            for nbr in set(graph.neighbors(edge[0])) & set(graph.neighbors(edge[1])):
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
