from __future__ import annotations

import collections
import itertools
import logging
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Optional,
    Sequence,
    Set,
    Union,
)

import networkx as nx
import numpy as np
from cytoolz import itertoolz
from spacy.tokens import Span, Token

from ... import errors
from . import utils as kt_utils

LOGGER = logging.getLogger(__name__)


def build_graph_from_terms(
    terms: Sequence[str] | Sequence[Token] | Sequence[Span],
    *,
    normalize: Optional[str | Callable[[Token], str]] = "lemma",
    window_size: int = 10,
    edge_weighting: str = "count",
) -> nx.Graph:
    """
    Transform an ordered list of non-overlapping terms into a graph,
    where each term is represented by a node with weighted edges linking it to
    other terms that co-occur within ``window_size`` terms of itself.

    Args:
        terms
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms;
            if falsy, use the form of terms as they appear in ``terms``;
            if a callable, must accept a ``Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.

            .. note:: This is applied to the elements of ``terms`` *only* if
               it's a list of ``Token`` or ``Span``.

        window_size: Size of sliding window over ``terms`` that determines
            which are said to co-occur. If 2, only immediately adjacent terms
            have edges in the returned network.
        edge_weighting ({"count", "binary"}): If "count", the nodes for
            all co-occurring terms are connected by edges with weight equal to
            the number of times they co-occurred within a sliding window;
            if "binary", all such edges have weight = 1.

    Returns:
        Networkx Graph whose nodes correspond to individual terms;
        those that co-occur are connected by edges with weights determined
        by ``edge_weighting``.
    """
    if window_size < 2:
        raise ValueError(f"window_size = {window_size} is invalid; value must be >= 2")
    if not terms:
        LOGGER.warning("input `terms` is empty, so output graph is also empty")
        return nx.Graph()

    # if len(terms) < window_size, cytoolz throws a StopIteration error; prevent it
    if len(terms) < window_size:
        LOGGER.info(
            "`terms` has fewer items (%s) than `window_size` (%s); "
            "setting window width to %s",
            len(terms),
            window_size,
            len(terms),
        )
        window_size = len(terms)

    first_term, terms = itertoolz.peek(terms)
    if isinstance(first_term, str):
        windows = itertoolz.sliding_window(window_size, terms)
    elif isinstance(first_term, (Span, Token)):
        windows = itertoolz.sliding_window(
            window_size, kt_utils.normalize_terms(terms, normalize)
        )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "terms",
                type(first_term),
                Union[Sequence[str], Sequence[Token], Sequence[Span]],
            )
        )

    graph = nx.Graph()
    if edge_weighting == "count":
        cooc_mat = collections.Counter(
            w1_w2
            for window in windows
            for w1_w2 in itertools.combinations(sorted(window), 2)
        )
        graph.add_edges_from(
            (w1, w2, {"weight": weight}) for (w1, w2), weight in cooc_mat.items()
        )
    elif edge_weighting == "binary":
        graph.add_edges_from(
            w1_w2 for window in windows for w1_w2 in itertools.combinations(window, 2)
        )
    else:
        raise ValueError(
            errors.value_invalid_msg(
                "edge_weighting", edge_weighting, {"count", "binary"}
            )
        )

    return graph


def rank_nodes_by_pagerank(
    graph: nx.Graph, weight: str = "weight", **kwargs,
) -> Dict[Any, float]:
    """
    Rank nodes in graph using the Pagegrank algorithm.

    Args:
        graph
        weight
        **kwargs

    Returns:
        Dict[object, float]
    """
    return nx.pagerank_scipy(graph, weight=weight, **kwargs)


def rank_nodes_by_bestcoverage(
    graph: nx.Graph, k: int, c: int = 1, alpha: float = 1.0, weight: str = "weight",
) -> Dict[Any, float]:
    """
    Rank nodes in a network using the BestCoverage algorithm that attempts to
    balance between node centrality and diversity.

    Args:
        graph
        k: Number of results to return for top-k search.
        c: *l* parameter for *l*-step expansion; best if 1 or 2
        alpha: Float in [0.0, 1.0] specifying how much of central vertex's score
            to remove from its *l*-step neighbors; smaller value puts more emphasis
            on centrality, larger value puts more emphasis on diversity
        weight: Key in edge data that holds weights.

    Returns:
        Top ``k`` nodes as ranked by bestcoverage algorithm; keys as node
        identifiers, values as corresponding ranking scores

    References:
        Küçüktunç, O., Saule, E., Kaya, K., & Çatalyürek, Ü. V. (2013, May).
        Diversified recommendation on graphs: pitfalls, measures, and algorithms.
        In Proceedings of the 22nd international conference on World Wide Web
        (pp. 715-726). International World Wide Web Conferences Steering Committee.
        http://www2013.wwwconference.org/proceedings/p715.pdf
    """
    alpha = float(alpha)

    nodes_list = [node for node in graph]
    if len(nodes_list) == 0:
        LOGGER.warning("`graph` is empty")
        return {}

    # ranks: array of PageRank values, summing up to 1
    ranks = nx.pagerank_scipy(graph, alpha=0.85, max_iter=100, tol=1e-08, weight=weight)
    sorted_ranks = sorted(ranks.items(), key=itemgetter(1), reverse=True)
    avg_degree = sum(dict(graph.degree()).values()) / len(nodes_list)
    # relaxation parameter, k' in the paper
    k_prime = int(k * avg_degree * c)

    top_k_sorted_ranks = sorted_ranks[:k_prime]

    def get_l_step_expanded_set(vertices: Collection[str], n_steps: int) -> Set[str]:
        """
        Args:
            vertices: vertices to be expanded
            n_steps: how many steps to expand vertices set

        Returns:
            the l-step expanded set of vertices
        """
        # add vertices to s
        s = set(vertices)
        # s.update(vertices)
        # for each step
        for _ in range(n_steps):
            # for each node
            next_vertices = []
            for vertex in vertices:
                # add its neighbors to the next list
                neighbors = graph.neighbors(vertex)
                next_vertices.extend(neighbors)
                s.update(neighbors)
            vertices = set(next_vertices)
        return s

    # TODO: someday, burton, figure out what you were going to do with this...
    top_k_exp_vertices = get_l_step_expanded_set(
        [item[0] for item in top_k_sorted_ranks], c
    )  # noqa: F841

    # compute initial exprel contribution
    taken = collections.defaultdict(bool)
    contrib = {}
    for vertex in nodes_list:
        # get l-step expanded set
        s = get_l_step_expanded_set([vertex], c)
        # sum up neighbors ranks, i.e. l-step expanded relevance
        contrib[vertex] = sum(ranks[v] for v in s)

    sum_contrib = 0.0
    results = {}
    # greedily select to maximize exprel metric
    for _ in range(k):
        if not contrib:
            break
        # find word with highest l-step expanded relevance score
        max_word_score = sorted(contrib.items(), key=itemgetter(1), reverse=True)[0]
        sum_contrib += max_word_score[1]  # contrib[max_word[0]]
        results[max_word_score[0]] = max_word_score[1]
        # find its l-step expanded set
        l_step_expanded_set = get_l_step_expanded_set([max_word_score[0]], c)
        # for each vertex found
        for vertex in l_step_expanded_set:
            # already removed its contribution from neighbors
            if taken[vertex] is True:
                continue
            # remove the contribution of vertex (or some fraction) from its l-step neighbors
            s1 = get_l_step_expanded_set([vertex], c)
            for w in s1:
                try:
                    contrib[w] -= alpha * ranks[vertex]
                except KeyError:
                    LOGGER.error(
                        "Word %s not in contrib dict! We're approximating...", w
                    )
            taken[vertex] = True
        contrib[max_word_score[0]] = 0

    return results


def rank_nodes_by_divrank(
    graph: nx.Graph,
    r: Optional[np.ndarray] = None,
    lambda_: float = 0.5,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Rank nodes in a network using the DivRank algorithm that attempts to
    balance between node centrality and diversity.

    Args:
        graph
        r: The "personalization vector"; by default, ``r = ones(1, n)/n``
        lambda_: Float in [0.0, 1.0]
        alpha: Float in [0.0, 1.0] that controls the strength of self-links.

    Returns:
        Mapping of node to score ordered by descending divrank score

    References:
        Mei, Q., Guo, J., & Radev, D. (2010, July). Divrank: the interplay of
        prestige and diversity in information networks. In Proceedings of the
        16th ACM SIGKDD international conference on Knowledge discovery and data
        mining (pp. 1009-1018). ACM.
        http://clair.si.umich.edu/~radev/papers/SIGKDD2010.pdf
    """
    # check function arguments
    if len(graph) == 0:
        LOGGER.warning("`graph` is empty")
        return {}

    # specify the order of nodes to use in creating the matrix
    # and then later recovering the values from the order index
    nodes_list = [node for node in graph]
    # create adjacency matrix, i.e.
    # n x n matrix where entry W_ij is the weight of the edge from V_i to V_j
    W = nx.to_numpy_matrix(graph, nodelist=nodes_list, weight="weight").A
    n = W.shape[1]
    # create flat prior personalization vector if none given
    if r is None:
        r = np.array([n * [1 / float(n)]])
    # Specify some constants
    max_iter = 1000
    diff = 1e10
    tol = 1e-3
    pr = np.array([n * [1 / float(n)]])
    # Get p0(v -> u), i.e. transition probability prior to reinforcement
    tmp = np.reshape(np.sum(W, axis=1), (n, 1))
    idx_nan = np.flatnonzero(tmp == 0)
    W0 = W / np.tile(tmp, (1, n))
    W0[idx_nan, :] = 0
    del W

    # DivRank algorithm
    i = 0
    while i < max_iter and diff > tol:
        W1 = alpha * W0 * np.tile(pr, (n, 1))
        W1 = W1 - np.diag(W1[:, 0]) + (1 - alpha) * np.diag(pr[0, :])
        tmp1 = np.reshape(np.sum(W1, axis=1), (n, 1))
        P = W1 / np.tile(tmp1, (1, n))
        P = ((1 - lambda_) * P) + (lambda_ * np.tile(r, (n, 1)))
        pr_new = np.dot(pr, P)
        i += 1
        diff = np.sum(np.abs(pr_new - pr)) / np.sum(pr)
        pr = pr_new

    # sort nodes by divrank score
    results = sorted(
        ((i, score) for i, score in enumerate(pr.flatten().tolist())),
        key=itemgetter(1),
        reverse=True,
    )

    # replace node number by node value
    divranks = {nodes_list[result[0]]: result[1] for result in results}

    return divranks
