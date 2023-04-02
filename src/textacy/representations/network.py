"""
Networks
--------

:mod:`textacy.representations.network`: Represent document data as networks,
where nodes are terms, sentences, or even full documents and edges between them
are weighted by the strength of their co-occurrence or similarity.
"""
from __future__ import annotations

import collections
import itertools
import logging
from operator import itemgetter
from typing import Any, Collection, Literal, Optional, Sequence, Union

import networkx as nx
import numpy as np
from cytoolz import itertoolz

from .. import errors, similarity


LOGGER = logging.getLogger(__name__)

try:
    nx_pagerank = nx.pagerank_scipy  # networkx < 3.0
except AttributeError:
    nx_pagerank = nx.pagerank  # networkx >= 3.0


def build_cooccurrence_network(
    data: Sequence[str] | Sequence[Sequence[str]],
    *,
    window_size: int = 2,
    edge_weighting: Literal["count", "binary"] = "count",
) -> nx.Graph:
    """
    Transform an ordered sequence of strings (or a sequence of such sequences)
    into a graph, where each string is represented by a node with weighted edges
    linking it to other strings that co-occur within ``window_size`` elements of itself.

    Input ``data`` can take a variety of forms. For example, as a ``Sequence[str]``
    where elements are token or term strings from a single document:

    .. code-block:: pycon

        >>> texts = [
        ...     "Mary had a little lamb. Its fleece was white as snow.",
        ...     "Everywhere that Mary went the lamb was sure to go.",
        ... ]
        >>> docs = [make_spacy_doc(text, lang="en_core_web_sm") for text in texts]
        >>> data = [tok.text for tok in docs[0]]
        >>> graph = build_cooccurrence_network(data, window_size=2)
        >>> sorted(graph.adjacency())[0]
        ('.', {'lamb': {'weight': 1}, 'Its': {'weight': 1}, 'snow': {'weight': 1}})

    Or as a ``Sequence[Sequence[str]]``, where elements are token or term strings
    per sentence from a single document:

    .. code-block:: pycon

        >>> data = [[tok.text for tok in sent] for sent in docs[0].sents]
        >>> graph = build_cooccurrence_network(data, window_size=2)
        >>> sorted(graph.adjacency())[0]
        ('.', {'lamb': {'weight': 1}, 'snow': {'weight': 1}})

    Or as a ``Sequence[Sequence[str]]``, where elements are token or term strings
    per document from multiple documents:

    .. code-block:: pycon

        >>> data = [[tok.text for tok in doc] for doc in docs]
        >>> graph = build_cooccurrence_network(data, window_size=2)
        >>> sorted(graph.adjacency())[0]
        ('.',
         {'lamb': {'weight': 1},
          'Its': {'weight': 1},
          'snow': {'weight': 1},
          'go': {'weight': 1}})

    Note how the "." token's connections to other nodes change for each case. (Note
    that in real usage, you'll probably want to remove stopwords, punctuation, etc.
    so that nodes in the graph represent meaningful concepts.)

    Args:
        data
        window_size: Size of sliding window over ``data`` that determines
            which strings are said to co-occur. For example, a value of 2 means that
            only immediately adjacent strings will have edges in the network;
            larger values loosen the definition of co-occurrence and typically
            lead to a more densely-connected network.

            .. note:: Co-occurrence windows are not permitted to cross sequences.
               So, if ``data`` is a ``Sequence[Sequence[str]]``, then co-occ counts
               are computed separately for each sub-sequence, then summed together.

        edge_weighting: Method by which edges between nodes are weighted.
            If "count", nodes are connected by edges with weights equal to
            the number of times they co-occurred within a sliding window;
            if "binary", all such edges have weight set equal to 1.

    Returns:
        Graph whose nodes correspond to individual strings from ``data``;
        those that co-occur are connected by edges with weights determined
        by ``edge_weighting``.

    Reference:
        https://en.wikipedia.org/wiki/Co-occurrence_network
    """
    if not data:
        LOGGER.warning("input `data` is empty, so output graph is also empty")
        return nx.Graph()

    if window_size < 2:
        raise ValueError(f"window_size = {window_size} is invalid; value must be >= 2")

    # input data is Sequence[str]
    if isinstance(data[0], str):
        windows = itertoolz.sliding_window(min(window_size, len(data)), data)
    # input data is Sequence[Sequence[str]]
    elif isinstance(data[0], Sequence) and isinstance(data[0][0], str):
        windows = itertoolz.concat(
            itertoolz.sliding_window(min(window_size, len(subseq)), subseq)
            for subseq in data
        )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "data", data, Union[Sequence[str], Sequence[Sequence[str]]]
            )
        )

    graph = nx.Graph()
    if edge_weighting == "count":
        cooc_counts = collections.Counter(
            w1_w2
            for window in windows
            for w1_w2 in itertools.combinations(sorted(window), 2)
        )
        graph.add_edges_from(
            (w1, w2, {"weight": weight}) for (w1, w2), weight in cooc_counts.items()
        )
    elif edge_weighting == "binary":
        edge_data = {"weight": 1}
        graph.add_edges_from(
            (w1, w2, edge_data)
            for window in windows
            for (w1, w2) in itertools.combinations(window, 2)
        )
    else:
        raise ValueError(
            errors.value_invalid_msg(
                "edge_weighting", edge_weighting, {"count", "binary"}
            )
        )

    return graph


def build_similarity_network(
    data: Sequence[str] | Sequence[Sequence[str]],
    edge_weighting: str,
) -> nx.Graph:
    """
    Transform a sequence of strings (or a sequence of such sequences) into a graph,
    where each element of the top-level sequence is represented by a node
    with edges linking it to all other elements weighted by their pairwise similarity.

    Input ``data`` can take a variety of forms. For example, as a ``Sequence[str]``
    where elements are sentence texts from a single document:

    .. code-block:: pycon

        >>> texts = [
        ...     "Mary had a little lamb. Its fleece was white as snow.",
        ...     "Everywhere that Mary went the lamb was sure to go.",
        ... ]
        >>> docs = [make_spacy_doc(text, lang="en_core_web_sm") for text in texts]
        >>> data = [sent.text.lower() for sent in docs[0].sents]
        >>> graph = build_similarity_network(data, "levenshtein")
        >>> sorted(graph.adjacency())[0]
        ('its fleece was white as snow.',
         {'mary had a little lamb.': {'weight': 0.24137931034482762}})

    Or as a ``Sequence[str]`` where elements are full texts from multiple documents:

    .. code-block:: pycon

        >>> data = [doc.text.lower() for doc in docs]
        >>> graph = build_similarity_network(data, "jaro")
        >>> sorted(graph.adjacency())[0]
        ('everywhere that mary went the lamb was sure to go.',
         {'mary had a little lamb. its fleece was white as snow.': {'weight': 0.6516002795248078}})

    Or as a ``Sequence[Sequence[str]]`` where elements are tokenized texts from
    multiple documents:

    .. code-block:: pycon

        >>> data = [[tok.lower_ for tok in doc] for doc in docs]
        >>> graph = build_similarity_network(data, "jaccard")
        >>> sorted(graph.adjacency())[0]
        (('everywhere', 'that', 'mary', 'went', 'the', 'lamb', 'was', 'sure', 'to', 'go', '.'),
         {('mary', 'had', 'a', 'little', 'lamb', '.', 'its', 'fleece', 'was', 'white', 'as', 'snow', '.'): {'weight': 0.21052631578947367}})

    Args:
        data
        edge_weighting: Similarity metric to use for weighting edges between elements
            in ``data``, represented as the name of a function available in
            :mod:`textacy.similarity`.

            .. note:: Different metrics are suited for different forms and contexts
               of ``data``. You'll have to decide which method makes sense. For example,
               when comparing a sequence of short strings, "levenshtein" is often
               a reasonable bet; when comparing a sequence of sequences of somewhat
               noisy strings (e.g. includes punctuation, cruft tokens), you might try
               "matching_subsequences_ratio" to help filter out the noise.

    Returns:
        Graph whose nodes correspond to top-level sequence elements in ``data``,
        connected by edges to all other nodes with weights determined by
        their pairwise similarity.

    Reference:
        https://en.wikipedia.org/wiki/Semantic_similarity_network -- this is *not*
        the same as what's implemented here, but they're similar in spirit.
    """
    if not data:
        LOGGER.warning("input `data` is empty, so output graph is also empty")
        return nx.Graph()

    sim_func = getattr(similarity, edge_weighting)

    if isinstance(data[0], str):
        ele_pairs = itertools.combinations(data, 2)
    elif isinstance(data[0], Sequence) and isinstance(data[0][0], str):
        # nx graph nodes need to be *hashable*, so Sequence => Tuple
        ele_pairs = (
            (tuple(ele1), tuple(ele2)) for ele1, ele2 in itertools.combinations(data, 2)
        )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "data", data, Union[Sequence[str], Sequence[Sequence[str]]]
            )
        )

    graph = nx.Graph()
    graph.add_edges_from(
        (ele1, ele2, {"weight": sim_func(ele1, ele2)}) for ele1, ele2 in ele_pairs
    )
    return graph


def rank_nodes_by_pagerank(
    graph: nx.Graph,
    weight: str = "weight",
    **kwargs,
) -> dict[Any, float]:
    """
    Rank nodes in ``graph`` using the Pagegrank algorithm.

    Args:
        graph
        weight: Key in edge data that holds weights.
        **kwargs

    Returns:
        Mapping of node object to Pagerank score.
    """
    return nx_pagerank(graph, weight=weight, **kwargs)


def rank_nodes_by_bestcoverage(
    graph: nx.Graph,
    k: int,
    c: int = 1,
    alpha: float = 1.0,
    weight: str = "weight",
) -> dict[Any, float]:
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
    ranks = nx_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-08, weight=weight)
    # sorted_ranks = sorted(ranks.items(), key=itemgetter(1), reverse=True)
    # avg_degree = sum(dict(graph.degree()).values()) / len(nodes_list)
    # relaxation parameter, k' in the paper
    # k_prime = int(k * avg_degree * c)

    # top_k_sorted_ranks = sorted_ranks[:k_prime]

    def get_l_step_expanded_set(vertices: Collection[str], n_steps: int) -> set[str]:
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
    # top_k_exp_vertices = get_l_step_expanded_set(
    #     [item[0] for item in top_k_sorted_ranks], c
    # )

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
) -> dict[str, float]:
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
    try:
        # networkx < 3.0
        W = nx.to_numpy_matrix(graph, nodelist=nodes_list, weight="weight").A
    except AttributeError:
        # networkx >= 3.0
        W = nx.adjacency_matrix(graph, nodelist=nodes_list, weight="weight").toarray()
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
