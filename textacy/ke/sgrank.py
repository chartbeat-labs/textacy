from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import math
import operator

import networkx as nx
from cytoolz import itertoolz

from .. import compat, extract


def sgrank(
    doc,
    ngrams=(1, 2, 3, 4, 5, 6),
    normalize="lemma",
    window_width=1500,
    n_keyterms=10,
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
        window_width (int): Width of sliding window in which term
            co-occurrences are determined to occur. Note: Larger values may
            dramatically increase runtime, owing to the larger number of
            co-occurrence combinations that must be counted.
        n_keyterms (int or float): Number of top-ranked terms to return as
            keyterms. If int, represents the absolute number; if float, must be
            in the open interval (0.0, 1.0), and is converted to an integer by
            ``int(round(len(doc) * n_keyterms))``
        idf (dict): Mapping of ``normalize(term)`` to inverse document frequency
            for re-weighting of unigrams (n-grams with n > 1 have df assumed = 1).
            Results are typically better with idf information.

    Returns:
        List[Tuple[str, float]]: Sorted list of top ``n_keyterms`` key terms and
        their corresponding SGRank scores

    Raises:
        ValueError: if ``n_keyterms`` is a float but not in (0.0, 1.0] or
            ``window_width`` < 2

    Reference:
        Danesh, Sumner, and Martin. "SGRank: Combining Statistical and Graphical
        Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction."
        Lexical and Computational Semantics (* SEM 2015) (2015): 117.
    """
    n_toks = len(doc)
    if isinstance(n_keyterms, float):
        if not 0.0 < n_keyterms <= 1.0:
            raise ValueError(
                "`n_keyterms` must be an int, or a float between 0.0 and 1.0"
            )
        n_keyterms = int(round(n_toks * n_keyterms))
    if window_width < 2:
        raise ValueError("`window_width` must be >= 2")
    window_width = min(n_toks, window_width)
    min_term_freq = min(n_toks // 1000, 4)
    if isinstance(ngrams, int):
        ngrams = (ngrams,)

    # build full list of candidate terms
    # if inverse doc freqs available, include nouns, adjectives, and verbs;
    # otherwise, just include nouns and adjectives
    # (without IDF downweighting, verbs dominate the results in a bad way)
    include_pos = {"NOUN", "PROPN", "ADJ", "VERB"} if idf else {"NOUN", "PROPN", "ADJ"}
    terms = itertoolz.concat(
        extract.ngrams(
            doc,
            n,
            filter_stops=True,
            filter_punct=True,
            filter_nums=False,
            include_pos=include_pos,
            min_freq=min_term_freq,
        )
        for n in ngrams
    )

    # get normalized term strings, as desired
    # paired with positional index in document and length in a 3-tuple
    if normalize == "lemma":
        terms = [(term.lemma_, term.start, len(term)) for term in terms]
    elif normalize == "lower":
        terms = [(term.lower_, term.start, len(term)) for term in terms]
    elif not normalize:
        terms = [(term.text, term.start, len(term)) for term in terms]
    else:
        terms = [(normalize(term), term.start, len(term)) for term in terms]

    # pre-filter terms to the top N ranked by TF or modified TF*IDF
    n_prefilter_kts = max(3 * n_keyterms, 100)
    term_text_counts = collections.Counter(term[0] for term in terms)
    if idf:
        mod_tfidfs = {
            term: count * idf.get(term, 1) if " " not in term else count
            for term, count in term_text_counts.items()
        }
        terms_set = {
            term
            for term, _ in sorted(
                mod_tfidfs.items(), key=operator.itemgetter(1), reverse=True
            )[:n_prefilter_kts]
        }
    else:
        terms_set = {term for term, _ in term_text_counts.most_common(n_prefilter_kts)}
    terms = [term for term in terms if term[0] in terms_set]

    # compute term weights from statistical attributes:
    # not subsumed frequency, position of first occurrence, and num words
    term_weights = {}
    seen_terms = set()
    n_toks_plus_1 = n_toks + 1
    for term in terms:
        term_text = term[0]
        # we only want the *first* occurrence of a unique term (by its text)
        if term_text in seen_terms:
            continue
        seen_terms.add(term_text)
        pos_first_occ_factor = math.log(n_toks_plus_1 / (term[1] + 1))
        # TODO: assess how best to scale term len
        term_len = math.sqrt(term[2])  # term[2]
        term_count = term_text_counts[term_text]
        subsum_count = sum(
            term_text_counts[t2]
            for t2 in terms_set
            if t2 != term_text and term_text in t2
        )
        term_freq_factor = term_count - subsum_count
        if idf and term[2] == 1:
            term_freq_factor *= idf.get(term_text, 1)
        term_weights[term_text] = term_freq_factor * pos_first_occ_factor * term_len

    # filter terms to only those with positive weights
    terms = [term for term in terms if term_weights[term[0]] > 0]

    n_coocs = collections.defaultdict(lambda: collections.defaultdict(int))
    sum_logdists = collections.defaultdict(lambda: collections.defaultdict(float))

    # iterate over windows
    log_ = math.log  # localize this, for performance
    for start_ind in compat.range_(n_toks):
        end_ind = start_ind + window_width
        window_terms = (term for term in terms if start_ind <= term[1] <= end_ind)
        # get all token combinations within window
        for t1, t2 in itertools.combinations(window_terms, 2):
            n_coocs[t1[0]][t2[0]] += 1
            sum_logdists[t1[0]][t2[0]] += log_(
                window_width / max(abs(t1[1] - t2[1]), 1)
            )
        if end_ind > n_toks:
            break

    # compute edge weights between co-occurring terms (nodes)
    edge_weights = collections.defaultdict(lambda: collections.defaultdict(float))
    for t1, t2s in sum_logdists.items():
        for t2 in t2s:
            edge_weights[t1][t2] = (
                ((1.0 + sum_logdists[t1][t2]) / n_coocs[t1][t2])
                * term_weights[t1]
                * term_weights[t2]
            )
    # normalize edge weights by sum of outgoing edge weights per term (node)
    norm_edge_weights = []
    for t1, t2s in edge_weights.items():
        sum_edge_weights = sum(t2s.values())
        norm_edge_weights.extend(
            (t1, t2, {"weight": weight / sum_edge_weights})
            for t2, weight in t2s.items()
        )

    # build the weighted directed graph from edges, rank nodes by pagerank
    graph = nx.DiGraph()
    graph.add_edges_from(norm_edge_weights)
    term_ranks = nx.pagerank_scipy(graph)

    return sorted(term_ranks.items(), key=operator.itemgetter(1, 0), reverse=True)[
        :n_keyterms
    ]
