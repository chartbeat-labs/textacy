from __future__ import annotations

import collections
from operator import itemgetter
from typing import Callable, Collection, Optional

from spacy.tokens import Doc, Token

from ... import representations, utils
from .. import utils as ext_utils


def textrank(
    doc: Doc,
    *,
    normalize: Optional[str | Callable[[Token], str]] = "lemma",
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    window_size: int = 2,
    edge_weighting: str = "binary",
    position_bias: bool = False,
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """
    Extract key terms from a document using the TextRank algorithm, or
    a variation thereof. For example:

    - TextRank: ``window_size=2, edge_weighting="binary", position_bias=False``
    - SingleRank: ``window_size=10, edge_weighting="count", position_bias=False``
    - PositionRank: ``window_size=10, edge_weighting="count", position_bias=True``

    Args:
        doc: spaCy ``Doc`` from which to extract keyterms.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms; if None,
            use the form of terms as they appeared in ``doc``; if a callable,
            must accept a ``Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        include_pos: One or more POS tags with which to filter for good candidate keyterms.
            If None, include tokens of all POS tags
            (which also allows keyterm extraction from docs without POS-tagging.)
        window_size: Size of sliding window in which term co-occurrences are determined.
        edge_weighting ({"count", "binary"}): : If "count", the nodes for
            all co-occurring terms are connected by edges with weight equal to
            the number of times they co-occurred within a sliding window;
            if "binary", all such edges have weight = 1.
        position_bias: If True, bias the PageRank algorithm for weighting
            nodes in the word graph, such that words appearing earlier and more
            frequently in ``doc`` tend to get larger weights.
        topn: Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(set(candidates)) * topn))``.

    Returns:
        Sorted list of top ``topn`` key terms and their corresponding TextRank ranking scores.

    References:
        - Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into texts.
          Association for Computational Linguistics.
        - Wan, Xiaojun and Jianguo Xiao. 2008. Single document keyphrase extraction
          using neighborhood knowledge. In Proceedings of the 23rd AAAI Conference
          on Artificial Intelligence, pages 855–860.
        - Florescu, C. and Cornelia, C. (2017). PositionRank: An Unsupervised Approach
          to Keyphrase Extraction from Scholarly Documents. In proceedings of ACL*,
          pages 1105-1115.
    """
    # validate / transform args
    include_pos = utils.to_set(include_pos) if include_pos else None
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                "topn={} is invalid; "
                "must be an int, or a float between 0.0 and 1.0".format(topn)
            )

    # bail out on empty docs
    if not doc:
        return []

    word_pos: Optional[dict[str, float]]
    if position_bias is True:
        word_pos = collections.defaultdict(float)
        for word, norm_word in zip(doc, ext_utils.terms_to_strings(doc, normalize)):
            word_pos[norm_word] += 1 / (word.i + 1)
        sum_word_pos = sum(word_pos.values())
        word_pos = {word: pos / sum_word_pos for word, pos in word_pos.items()}
    else:
        word_pos = None
    # build a graph from all words in doc, then score them
    graph = representations.network.build_cooccurrence_network(
        list(ext_utils.terms_to_strings(doc, normalize)),
        window_size=window_size,
        edge_weighting=edge_weighting,
    )
    word_scores = representations.network.rank_nodes_by_pagerank(
        graph, weight="weight", personalization=word_pos
    )
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


def _get_candidates(
    doc: Doc,
    normalize: Optional[str | Callable],
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
            include_pos is None or tok.pos_ in include_pos
        )

    candidates = ext_utils.get_longest_subsequence_candidates(doc, _is_valid_tok)
    return {
        tuple(ext_utils.terms_to_strings(candidate, normalize))  # type: ignore
        for candidate in candidates
    }
