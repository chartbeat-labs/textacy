from __future__ import annotations

import collections
import itertools
import math
from operator import itemgetter
from typing import Callable, Collection, Counter, Optional

import networkx as nx
from spacy.tokens import Doc, Span

from ... import utils
from .. import utils as ext_utils


try:
    nx_pagerank = nx.pagerank_scipy  # networkx < 3.0
except AttributeError:
    nx_pagerank = nx.pagerank  # networkx >= 3.0


Candidate = collections.namedtuple("Candidate", ["text", "idx", "length", "count"])


def sgrank(
    doc: Doc,
    *,
    normalize: Optional[str | Callable[[Span], str]] = "lemma",
    ngrams: int | Collection[int] = (1, 2, 3, 4, 5, 6),
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    window_size: int = 1500,
    topn: int | float = 10,
    idf: Optional[dict[str, float]] = None,
) -> list[tuple[str, float]]:
    """
    Extract key terms from a document using the SGRank algorithm.

    Args:
        doc: spaCy ``Doc`` from which to extract keyterms.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms; if None,
            use the form of terms as they appeared in ``doc``; if a callable,
            must accept a ``Span`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        ngrams: n of which n-grams to include. For example, ``(1, 2, 3, 4, 5, 6)`` (default)
            includes all ngrams from 1 to 6; `2` if only bigrams are wanted
        include_pos: One or more POS tags with which to filter for good candidate keyterms.
            If None, include tokens of all POS tags
            (which also allows keyterm extraction from docs without POS-tagging.)
        window_size: Size of sliding window in which term co-occurrences are determined
            to occur. Note: Larger values may dramatically increase runtime, owing to
            the larger number of co-occurrence combinations that must be counted.
        topn: Number of top-ranked terms to return as keyterms.
            If int, represents the absolute number; if float, must be in the open interval
            (0.0, 1.0), and is converted to an integer by ``int(round(len(candidates) * topn))``
        idf: Mapping of ``normalize(term)`` to inverse document frequency
            for re-weighting of unigrams (n-grams with n > 1 have df assumed = 1).
            Results are typically better with idf information.

    Returns:
        Sorted list of top ``topn`` key terms and their corresponding SGRank scores

    Raises:
        ValueError: if ``topn`` is a float but not in (0.0, 1.0] or ``window_size`` < 2

    References:
        Danesh, Sumner, and Martin. "SGRank: Combining Statistical and Graphical
        Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction."
        Lexical and Computational Semantics (* SEM 2015) (2015): 117.
    """
    # validate / transform args
    ngrams: tuple[int, ...] = utils.to_tuple(ngrams)
    include_pos: Optional[set[str]] = utils.to_set(include_pos) if include_pos else None
    if window_size < 2:
        raise ValueError("`window_size` must be >= 2")
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError("`topn` must be an int, or a float between 0.0 and 1.0")

    n_toks = len(doc)
    window_size = min(n_toks, window_size)
    # bail out on (nearly) empty docs
    if n_toks < 2:
        return []

    candidates, candidate_counts = _get_candidates(doc, normalize, ngrams, include_pos)
    # scale float topn based on total number of initial candidates
    if isinstance(topn, float):
        topn = int(round(len(candidate_counts) * topn))
    candidates, unique_candidates = _prefilter_candidates(
        candidates, candidate_counts, topn, idf
    )

    term_weights = _compute_term_weights(
        candidates, candidate_counts, unique_candidates, n_toks, idf
    )
    # filter terms to only those with positive weights
    candidates = [cand for cand in candidates if term_weights[cand.text] > 0]
    edge_weights = _compute_edge_weights(candidates, term_weights, window_size, n_toks)

    # build the weighted directed graph from edges, rank nodes by pagerank
    graph = nx.DiGraph()
    graph.add_edges_from(edge_weights)
    term_ranks = nx_pagerank(graph, alpha=0.85, weight="weight")
    sorted_term_ranks = sorted(term_ranks.items(), key=itemgetter(1, 0), reverse=True)

    return ext_utils.get_filtered_topn_terms(
        sorted_term_ranks, topn, match_threshold=0.8
    )


def _get_candidates(
    doc: Doc,
    normalize: Optional[str | Callable[[Span], str]],
    ngrams: tuple[int, ...],
    include_pos: Optional[set[str]],
) -> tuple[list[Candidate], Counter[str]]:
    """
    Get n-gram candidate keyterms from ``doc``, with key information for each:
    its normalized text string, position within the doc, number of constituent words,
    and frequency of occurrence.
    """
    min_term_freq = min(max(len(doc) // 1000, 1), 4)
    cand_tuples = list(
        ext_utils.get_ngram_candidates(doc, ngrams, include_pos=include_pos)
    )
    cand_texts = [
        " ".join(ext_utils.terms_to_strings(ctup, normalize or "orth"))
        for ctup in cand_tuples
    ]
    cand_counts = collections.Counter(cand_texts)
    candidates = [
        Candidate(text=ctext, idx=ctup[0].i, length=len(ctup), count=cand_counts[ctext])
        for ctup, ctext in zip(cand_tuples, cand_texts)
    ]
    if min_term_freq > 1:
        candidates = [
            candidate for candidate in candidates if candidate.count >= min_term_freq
        ]
    return candidates, cand_counts


def _prefilter_candidates(
    candidates: list[Candidate],
    candidate_counts: Counter[str],
    topn: int,
    idf: Optional[dict[str, float]],
) -> tuple[list[Candidate], set[str]]:
    """
    Filter initial set of candidates to only those with sufficiently high TF or
    (if available) modified TF*IDF.
    """
    topn_prefilter = max(3 * topn, 100)
    if idf:
        mod_tfidfs = {
            c.text: c.count * idf.get(c.text, 1) if " " not in c.text else c.count
            for c in candidates
        }
        unique_candidates = {
            ctext
            for ctext, _ in sorted(mod_tfidfs.items(), key=itemgetter(1), reverse=True)[
                :topn_prefilter
            ]
        }
    else:
        unique_candidates = {
            ctext for ctext, _ in candidate_counts.most_common(topn_prefilter)
        }
    candidates = [cand for cand in candidates if cand.text in unique_candidates]
    return candidates, unique_candidates


def _compute_term_weights(
    candidates: list[Candidate],
    candidate_counts: dict[str, int],
    unique_candidates: set[str],
    n_toks: int,
    idf: Optional[dict[str, float]],
) -> dict[str, float]:
    """
    Compute term weights from statistical attributes: position of first occurrence,
    not subsumed frequency, and number of constituent words.
    """
    clen: float
    term_weights = {}
    seen_terms = set()
    n_toks_p1 = n_toks + 1
    for cand in candidates:
        # we only want the *first* occurrence of a unique term (by its text)
        if cand.text in seen_terms:
            continue
        else:
            seen_terms.add(cand.text)
        pos_first_occ = math.log(n_toks_p1 / (cand.idx + 1))
        # TODO: do we want to sub-linear scale term len or not?
        clen = math.sqrt(cand.length)
        # subtract from subsum_count for the case when ctext == ctext2
        subsum_count = (
            sum(
                candidate_counts[ctext2]
                for ctext2 in unique_candidates
                if cand.text in ctext2
            )
            - cand.count
        )
        term_freq_factor = cand.count - subsum_count
        if idf and clen == 1:
            term_freq_factor *= idf.get(cand.text, 1)
        term_weights[cand.text] = term_freq_factor * pos_first_occ * clen
    return term_weights


def _compute_edge_weights(
    candidates: list[Candidate],
    term_weights: dict[str, float],
    window_size: int,
    n_toks: int,
) -> list[tuple[str, str, dict[str, float]]]:
    """
    Compute weights between candidates that occur within a sliding window(s) of
    each other, then combine with statistical ``term_weights`` and normalize
    by the total number of outgoing edge weights.
    """
    n_coocs: collections.defaultdict = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    sum_logdists: collections.defaultdict = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    # iterate over windows
    log_ = math.log  # localize this, for performance
    for start_idx in range(n_toks):
        end_idx = start_idx + window_size
        window_cands = (cand for cand in candidates if start_idx <= cand.idx < end_idx)
        # get all token combinations within window
        for c1, c2 in itertools.combinations(window_cands, 2):
            n_coocs[c1.text][c2.text] += 1
            try:
                sum_logdists[c1.text][c2.text] += log_(
                    window_size / abs(c1.idx - c2.idx)
                )
            except ZeroDivisionError:
                sum_logdists[c1.text][c2.text] += log_(window_size)
        if end_idx >= n_toks:
            break
    # compute edge weights between co-occurring terms (nodes)
    edge_weights: collections.defaultdict = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for c1, c2_dict in sum_logdists.items():
        for c2, sum_logdist in c2_dict.items():
            edge_weights[c1][c2] = (
                ((1.0 + sum_logdist) / n_coocs[c1][c2])
                * term_weights[c1]
                * term_weights[c2]
            )
    # normalize edge weights by sum of outgoing edge weights per term (node)
    norm_edge_weights: list[tuple[str, str, dict[str, float]]] = []
    for c1, c2s in edge_weights.items():
        sum_edge_weights = sum(c2s.values())
        norm_edge_weights.extend(
            (c1, c2, {"weight": weight / sum_edge_weights})
            for c2, weight in c2s.items()
        )
    return norm_edge_weights
