# -*- coding: utf-8 -*-
"""
Functions for unsupervised automatic key term extraction, both specific algorithms
(SGRank, TextRank, SingleRank) and a generalization of semantic network-based methods.
Also includes a function to aggregate common key term variants of the same idea.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import logging
import math
import operator
from decimal import Decimal

import networkx as nx
import numpy as np
from cytoolz import itertoolz

from . import compat
from . import extract
from . import network
from . import similarity
from . import vsm

LOGGER = logging.getLogger(__name__)


def sgrank(
    doc,
    ngrams=(1, 2, 3, 4, 5, 6),
    normalize="lemma",
    window_width=1500,
    n_keyterms=10,
    idf=None,
):
    """
    Extract key terms from a document using the [SGRank]_ algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3, 4, 5, 6)``
                (default) includes all ngrams from 1 to 6; `2`
                if only bigrams are wanted
        normalize (str or callable): If 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``spacy.Span`` and return a str,
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
        List[Tuple[str, float]]: sorted list of top ``n_keyterms`` key terms and
        their corresponding SGRank scores

    Raises:
        ValueError: If ``n_keyterms`` is a float but not in (0.0, 1.0] or
            ``window_width`` < 2.

    References:
        .. [SGRank] Danesh, Sumner, and Martin. "SGRank: Combining Statistical and
           Graphical Methods to Improve the State of the Art in Unsupervised Keyphrase
           Extraction". Lexical and Computational Semantics (* SEM 2015) (2015): 117.
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


def textrank(doc, normalize="lemma", n_keyterms=10):
    """
    Convenience function for calling :func:`key_terms_from_semantic_network <textacy.keyterms.key_terms_from_semantic_network>`
    with the parameter values used in the [TextRank]_ algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``spacy.Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        n_keyterms (int or float): if int, number of top-ranked terms
            to return as keyterms; if float, must be in the open interval (0, 1),
            representing the fraction of top-ranked terms to return as keyterms

    Returns:
        See :func:`key_terms_from_semantic_network`.

    References:
        .. [TextRank] Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing
           order into texts. Association for Computational Linguistics.
    """
    return key_terms_from_semantic_network(
        doc,
        normalize=normalize,
        window_width=2,
        edge_weighting="binary",
        ranking_algo="pagerank",
        join_key_words=False,
        n_keyterms=n_keyterms,
    )


def singlerank(doc, normalize="lemma", n_keyterms=10):
    """
    Convenience function for calling :func:`key_terms_from_semantic_network <textacy.keyterms.key_terms_from_semantic_network>`
    with the parameter values used in the [SingleRank]_ algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``spacy.Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        n_keyterms (int or float): if int, number of top-ranked terms
            to return as keyterms; if float, must be in the open interval (0, 1),
            representing the fraction of top-ranked terms to return as keyterms

    Returns:
        see :func:`key_terms_from_semantic_network`.

    References:
        .. [SingleRank] Hasan, K. S., & Ng, V. (2010, August). Conundrums in unsupervised
           keyphrase extraction: making sense of the state-of-the-art. In Proceedings
           of the 23rd International Conference on Computational Linguistics:
           Posters (pp. 365-373). Association for Computational Linguistics.
    """
    return key_terms_from_semantic_network(
        doc,
        normalize=normalize,
        window_width=10,
        edge_weighting="cooc_freq",
        ranking_algo="pagerank",
        join_key_words=True,
        n_keyterms=n_keyterms,
    )


def key_terms_from_semantic_network(
    doc,
    normalize="lemma",
    window_width=2,
    edge_weighting="binary",
    ranking_algo="pagerank",
    join_key_words=False,
    n_keyterms=10,
    **kwargs
):
    """
    Extract key terms from a document by ranking nodes in a semantic network of
    terms, connected by edges and weights specified by parameters.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if None, use the form of terms as they appeared in
            ``doc``; if a callable, must accept a ``spacy.Token`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        window_width (int): width of sliding window in which term
            co-occurrences are said to occur
        edge_weighting ('binary', 'cooc_freq'}): method used to
            determine weights of edges between nodes in the semantic network;
            if 'binary', edge weight is set to 1 for any two terms co-occurring
            within `window_width` terms; if 'cooc_freq', edge weight is set to
            the number of times that any two terms co-occur
        ranking_algo ({'pagerank', 'divrank', 'bestcoverage'}):
            algorithm with which to rank nodes in the semantic network;
            `pagerank` is the canonical (and default) algorithm, but it prioritizes
            node centrality at the expense of node diversity; the other two
            attempt to balance centrality with diversity
        join_key_words (bool): if True, join consecutive key words
            together into longer key terms, taking the sum of the constituent words'
            scores as the joined key term's combined score
        n_keyterms (int or float): if int, number of top-ranked terms
            to return as keyterms; if float, must be in the open interval (0, 1),
            is converted to an integer by ``round(len(doc) * n_keyterms)``

    Returns:
        List[Tuple[str, float]]: sorted list of top ``n_keyterms`` key terms and
        their corresponding ranking scores

    Raises:
        ValueError: if ``n_keyterms`` is a float but not in (0.0, 1.0]
    """
    if isinstance(n_keyterms, float):
        if not 0.0 < n_keyterms <= 1.0:
            raise ValueError(
                "`n_keyterms` must be an int, or a float between 0.0 and 1.0"
            )
        n_keyterms = int(round(len(doc) * n_keyterms))

    include_pos = {"NOUN", "PROPN", "ADJ"}
    if normalize == "lemma":
        word_list = [word.lemma_ for word in doc]
        good_word_list = [
            word.lemma_
            for word in doc
            if not word.is_stop and not word.is_punct and word.pos_ in include_pos
        ]
    elif normalize == "lower":
        word_list = [word.lower_ for word in doc]
        good_word_list = [
            word.lower_
            for word in doc
            if not word.is_stop and not word.is_punct and word.pos_ in include_pos
        ]
    elif not normalize:
        word_list = [word.text for word in doc]
        good_word_list = [
            word.text
            for word in doc
            if not word.is_stop and not word.is_punct and word.pos_ in include_pos
        ]
    else:
        word_list = [normalize(word) for word in doc]
        good_word_list = [
            normalize(word)
            for word in doc
            if not word.is_stop and not word.is_punct and word.pos_ in include_pos
        ]

    # HACK: omit empty strings, which happen as a bug in spacy as of v1.5
    # and may well happen with ``normalize`` as a callable
    # an empty string should never be considered a keyterm
    good_word_list = [word for word in good_word_list if word]
    graph = network.terms_to_semantic_network(
        good_word_list, window_width=window_width, edge_weighting=edge_weighting
    )

    # rank nodes by algorithm, and sort in descending order
    if ranking_algo == "pagerank":
        word_ranks = nx.pagerank_scipy(graph, weight="weight")
    elif ranking_algo == "divrank":
        word_ranks = rank_nodes_by_divrank(
            graph,
            r=None,
            lambda_=kwargs.get("lambda_", 0.5),
            alpha=kwargs.get("alpha", 0.5),
        )
    elif ranking_algo == "bestcoverage":
        word_ranks = rank_nodes_by_bestcoverage(
            graph, k=n_keyterms, c=kwargs.get("c", 1), alpha=kwargs.get("alpha", 1.0)
        )

    # bail out here if all we wanted was key *words* and not *terms*
    if join_key_words is False:
        return [
            (word, score)
            for word, score in sorted(
                word_ranks.items(), key=operator.itemgetter(1), reverse=True
            )[:n_keyterms]
        ]

    top_n = int(0.25 * len(word_ranks))
    top_word_ranks = {
        word: rank
        for word, rank in sorted(
            word_ranks.items(), key=operator.itemgetter(1), reverse=True
        )[:top_n]
    }

    # join consecutive key words into key terms
    seen_joined_key_terms = set()
    joined_key_terms = []
    for key, group in itertools.groupby(word_list, lambda word: word in top_word_ranks):
        if key is True:
            words = list(group)
            term = " ".join(words)
            if term in seen_joined_key_terms:
                continue
            seen_joined_key_terms.add(term)
            joined_key_terms.append((term, sum(word_ranks[word] for word in words)))

    return sorted(joined_key_terms, key=operator.itemgetter(1, 0), reverse=True)[
        :n_keyterms
    ]


def most_discriminating_terms(
    terms_lists, bool_array_grp1, max_n_terms=1000, top_n_terms=25
):
    """
    Given a collection of documents assigned to 1 of 2 exclusive groups, get the
    `top_n_terms` most discriminating terms for group1-and-not-group2 and
    group2-and-not-group1.

    Args:
        terms_lists (Iterable[Iterable[str]]): a sequence of documents, each as a
            sequence of (str) terms; used as input to :func:`doc_term_matrix()`
        bool_array_grp1 (Iterable[bool]): an ordered sequence of True/False values,
            where True corresponds to documents falling into "group 1" and False
            corresponds to those in "group 2"
        max_n_terms (int): only consider terms whose document frequency is within
            the top `max_n_terms` out of all distinct terms; must be > 0
        top_n_terms (int or float): if int (must be > 0), the total number of most
            discriminating terms to return for each group; if float (must be in
            the interval (0, 1)), the fraction of `max_n_terms` to return for each group

    Returns:
        List[str]: top `top_n_terms` most discriminating terms for grp1-not-grp2

        List[str]: top `top_n_terms` most discriminating terms for grp2-not-grp1

    References:
        King, Gary, Patrick Lam, and Margaret Roberts. "Computer-Assisted Keyword
        and Document Set Discovery from Unstructured Text." (2014).
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.1445&rep=rep1&type=pdf
    """
    alpha_grp1 = 1
    alpha_grp2 = 1
    if isinstance(top_n_terms, float):
        top_n_terms = top_n_terms * max_n_terms
    bool_array_grp1 = np.array(bool_array_grp1)
    bool_array_grp2 = np.invert(bool_array_grp1)

    vectorizer = vsm.Vectorizer(
        tf_type="linear",
        norm=None,
        idf_type="smooth",
        min_df=3,
        max_df=0.95,
        max_n_terms=max_n_terms,
    )
    dtm = vectorizer.fit_transform(terms_lists)
    id2term = vectorizer.id_to_term

    # get doc freqs for all terms in grp1 documents
    dtm_grp1 = dtm[bool_array_grp1, :]
    n_docs_grp1 = dtm_grp1.shape[0]
    doc_freqs_grp1 = vsm.get_doc_freqs(dtm_grp1)

    # get doc freqs for all terms in grp2 documents
    dtm_grp2 = dtm[bool_array_grp2, :]
    n_docs_grp2 = dtm_grp2.shape[0]
    doc_freqs_grp2 = vsm.get_doc_freqs(dtm_grp2)

    # get terms that occur in a larger fraction of grp1 docs than grp2 docs
    term_ids_grp1 = np.where(
        doc_freqs_grp1 / n_docs_grp1 > doc_freqs_grp2 / n_docs_grp2
    )[0]

    # get terms that occur in a larger fraction of grp2 docs than grp1 docs
    term_ids_grp2 = np.where(
        doc_freqs_grp1 / n_docs_grp1 < doc_freqs_grp2 / n_docs_grp2
    )[0]

    # get grp1 terms doc freqs in and not-in grp1 and grp2 docs, plus marginal totals
    grp1_terms_grp1_df = doc_freqs_grp1[term_ids_grp1]
    grp1_terms_grp2_df = doc_freqs_grp2[term_ids_grp1]
    # grp1_terms_grp1_not_df = n_docs_grp1 - grp1_terms_grp1_df
    # grp1_terms_grp2_not_df = n_docs_grp2 - grp1_terms_grp2_df
    # grp1_terms_total_df = grp1_terms_grp1_df + grp1_terms_grp2_df
    # grp1_terms_total_not_df = grp1_terms_grp1_not_df + grp1_terms_grp2_not_df

    # get grp2 terms doc freqs in and not-in grp2 and grp1 docs, plus marginal totals
    grp2_terms_grp2_df = doc_freqs_grp2[term_ids_grp2]
    grp2_terms_grp1_df = doc_freqs_grp1[term_ids_grp2]
    # grp2_terms_grp2_not_df = n_docs_grp2 - grp2_terms_grp2_df
    # grp2_terms_grp1_not_df = n_docs_grp1 - grp2_terms_grp1_df
    # grp2_terms_total_df = grp2_terms_grp2_df + grp2_terms_grp1_df
    # grp2_terms_total_not_df = grp2_terms_grp2_not_df + grp2_terms_grp1_not_df

    # get grp1 terms likelihoods, then sort for most discriminating grp1-not-grp2 terms
    grp1_terms_likelihoods = {}
    for idx, term_id in enumerate(term_ids_grp1):
        term1 = (
            Decimal(math.factorial(grp1_terms_grp1_df[idx] + alpha_grp1 - 1))
            * Decimal(math.factorial(grp1_terms_grp2_df[idx] + alpha_grp2 - 1))
            / Decimal(
                math.factorial(
                    grp1_terms_grp1_df[idx]
                    + grp1_terms_grp2_df[idx]
                    + alpha_grp1
                    + alpha_grp2
                    - 1
                )
            )
        )
        term2 = (
            Decimal(
                math.factorial(n_docs_grp1 - grp1_terms_grp1_df[idx] + alpha_grp1 - 1)
            )
            * Decimal(
                math.factorial(n_docs_grp2 - grp1_terms_grp2_df[idx] + alpha_grp2 - 1)
            )
            / Decimal(
                (
                    math.factorial(
                        n_docs_grp1
                        + n_docs_grp2
                        - grp1_terms_grp1_df[idx]
                        - grp1_terms_grp2_df[idx]
                        + alpha_grp1
                        + alpha_grp2
                        - 1
                    )
                )
            )
        )
        grp1_terms_likelihoods[id2term[term_id]] = term1 * term2
    top_grp1_terms = [
        term
        for term, likelihood in sorted(
            grp1_terms_likelihoods.items(), key=operator.itemgetter(1), reverse=True
        )[:top_n_terms]
    ]

    # get grp2 terms likelihoods, then sort for most discriminating grp2-not-grp1 terms
    grp2_terms_likelihoods = {}
    for idx, term_id in enumerate(term_ids_grp2):
        term1 = (
            Decimal(math.factorial(grp2_terms_grp2_df[idx] + alpha_grp2 - 1))
            * Decimal(math.factorial(grp2_terms_grp1_df[idx] + alpha_grp1 - 1))
            / Decimal(
                math.factorial(
                    grp2_terms_grp2_df[idx]
                    + grp2_terms_grp1_df[idx]
                    + alpha_grp2
                    + alpha_grp1
                    - 1
                )
            )
        )
        term2 = (
            Decimal(
                math.factorial(n_docs_grp2 - grp2_terms_grp2_df[idx] + alpha_grp2 - 1)
            )
            * Decimal(
                math.factorial(n_docs_grp1 - grp2_terms_grp1_df[idx] + alpha_grp1 - 1)
            )
            / Decimal(
                (
                    math.factorial(
                        n_docs_grp2
                        + n_docs_grp1
                        - grp2_terms_grp2_df[idx]
                        - grp2_terms_grp1_df[idx]
                        + alpha_grp2
                        + alpha_grp1
                        - 1
                    )
                )
            )
        )
        grp2_terms_likelihoods[id2term[term_id]] = term1 * term2
    top_grp2_terms = [
        term
        for term, likelihood in sorted(
            grp2_terms_likelihoods.items(), key=operator.itemgetter(1), reverse=True
        )[:top_n_terms]
    ]

    return (top_grp1_terms, top_grp2_terms)


def aggregate_term_variants(terms, acro_defs=None, fuzzy_dedupe=True):
    """
    Take a set of unique terms and aggregate terms that are symbolic, lexical,
    and ordering variants of each other, as well as acronyms and fuzzy string matches.

    Args:
        terms (Set[str]): set of unique terms with potential duplicates
        acro_defs (dict): if not None, terms that are acronyms will be
            aggregated with their definitions and terms that are definitions will
            be aggregated with their acronyms
        fuzzy_dedupe (bool): if True, fuzzy string matching will be used
            to aggregate similar terms of a sufficient length

    Returns:
        List[Set[str]]: each item is a set of aggregated terms

    Notes:
        Partly inspired by aggregation of variants discussed in
        Park, Youngja, Roy J. Byrd, and Branimir K. Boguraev.
        "Automatic glossary extraction: beyond terminology identification."
        Proceedings of the 19th international conference on Computational linguistics-Volume 1.
        Association for Computational Linguistics, 2002.
    """
    agg_terms = []
    seen_terms = set()
    for term in sorted(terms, key=len, reverse=True):

        if term in seen_terms:
            continue

        variants = set([term])
        seen_terms.add(term)

        # symbolic variations
        if "-" in term:
            variant = term.replace("-", " ").strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)
        if "/" in term:
            variant = term.replace("/", " ").strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)

        # lexical variations
        term_words = term.split()
        # last_word = term_words[-1]
        # # assume last word is a noun
        # last_word_lemmatized = lemmatizer.lemmatize(last_word, 'n')
        # # if the same, either already a lemmatized noun OR a verb; try verb
        # if last_word_lemmatized == last_word:
        #     last_word_lemmatized = lemmatizer.lemmatize(last_word, 'v')
        # # if at least we have a new term... add it
        # if last_word_lemmatized != last_word:
        #     term_lemmatized = ' '.join(term_words[:-1] + [last_word_lemmatized])
        #     if term_lemmatized in terms.difference(seen_terms):
        #         variants.add(term_lemmatized)
        #         seen_terms.add(term_lemmatized)

        # if term is an acronym, add its definition
        # if term is a definition, add its acronym
        if acro_defs:
            for acro, def_ in acro_defs.items():
                if acro.lower() == term.lower():
                    variants.add(def_.lower())
                    seen_terms.add(def_.lower())
                    break
                elif def_.lower() == term.lower():
                    variants.add(acro.lower())
                    seen_terms.add(acro.lower())
                    break

        # if 3+ -word term differs by one word at the start or the end
        # of a longer phrase, aggregate
        if len(term_words) > 2:
            term_minus_first_word = " ".join(term_words[1:])
            term_minus_last_word = " ".join(term_words[:-1])
            if term_minus_first_word in terms.difference(seen_terms):
                variants.add(term_minus_first_word)
                seen_terms.add(term_minus_first_word)
            if term_minus_last_word in terms.difference(seen_terms):
                variants.add(term_minus_last_word)
                seen_terms.add(term_minus_last_word)
            # check for "X of Y" <=> "Y X" term variants
            if " of " in term:
                split_term = term.split(" of ")
                variant = split_term[1] + " " + split_term[0]
                if variant in terms.difference(seen_terms):
                    variants.add(variant)
                    seen_terms.add(variant)

        # intense de-duping for sufficiently long terms
        if fuzzy_dedupe is True and len(term) >= 13:
            for other_term in sorted(
                terms.difference(seen_terms), key=len, reverse=True
            ):
                if len(other_term) < 13:
                    break
                tsr = similarity.token_sort_ratio(term, other_term)
                if tsr > 0.93:
                    variants.add(other_term)
                    seen_terms.add(other_term)
                    break

        agg_terms.append(variants)

    return agg_terms


def rank_nodes_by_bestcoverage(graph, k, c=1, alpha=1.0):
    """
    Rank nodes in a network using the [BestCoverage]_ algorithm that attempts to
    balance between node centrality and diversity.

    Args:
        graph (:class:`networkx.Graph <networkx.Graph>`)
        k (int): number of results to return for top-k search
        c (int): *l* parameter for *l*-step expansion; best if 1 or 2
        alpha (float): float in [0.0, 1.0] specifying how much of
            central vertex's score to remove from its *l*-step neighbors;
            smaller value puts more emphasis on centrality, larger value puts
            more emphasis on diversity

    Returns:
        dict: top ``k`` nodes as ranked by bestcoverage algorithm; keys as node
        identifiers, values as corresponding ranking scores

    References:
        .. [BestCoverage] Küçüktunç, O., Saule, E., Kaya, K., & Çatalyürek, Ü. V.
           (2013, May). Diversified recommendation on graphs: pitfalls, measures,
           and algorithms. In Proceedings of the 22nd international conference on
           World Wide Web (pp. 715-726). International World Wide Web Conferences
           Steering Committee. http://www2013.wwwconference.org/proceedings/p715.pdf
    """
    alpha = float(alpha)

    nodes_list = [node for node in graph]
    if len(nodes_list) == 0:
        LOGGER.warning("``graph`` is empty!")
        return {}

    # ranks: array of PageRank values, summing up to 1
    ranks = nx.pagerank_scipy(
        graph, alpha=0.85, max_iter=100, tol=1e-08, weight="weight"
    )
    sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)
    avg_degree = sum(dict(graph.degree()).values()) / len(nodes_list)
    # relaxation parameter, k' in the paper
    k_prime = int(k * avg_degree * c)

    top_k_sorted_ranks = sorted_ranks[:k_prime]

    def get_l_step_expanded_set(vertices, l):
        """
        Args:
            vertices (iterable[str]): vertices to be expanded
            l (int): how many steps to expand vertices set

        Returns:
            set: the l-step expanded set of vertices
        """
        # add vertices to s
        s = set(vertices)
        # s.update(vertices)
        # for each step
        for _ in compat.range_(l):
            # for each node
            next_vertices = []
            for vertex in vertices:
                # add its neighbors to the next list
                neighbors = graph.neighbors(vertex)
                next_vertices.extend(neighbors)
                s.update(neighbors)
            vertices = set(next_vertices)
        return s

    top_k_exp_vertices = get_l_step_expanded_set(
        [item[0] for item in top_k_sorted_ranks], c
    )

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
    for _ in compat.range_(k):
        if not contrib:
            break
        # find word with highest l-step expanded relevance score
        max_word_score = sorted(
            contrib.items(), key=operator.itemgetter(1), reverse=True
        )[0]
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


def rank_nodes_by_divrank(graph, r=None, lambda_=0.5, alpha=0.5):
    """
    Rank nodes in a network using the [DivRank]_ algorithm that attempts to
    balance between node centrality and diversity.

    Args:
        graph (:class:`networkx.Graph <networkx.Graph>`):
        r (:class:`numpy.array`,): the "personalization vector";
            by default, ``r = ones(1, n)/n``
        lambda_ (float): must be in [0.0, 1.0]
        alpha (float): controls the strength of self-links;
            must be in [0.0, 1.0]

    Returns:
        List[Tuple[str, float]]: list of (node, score) tuples ordered by desc. divrank score

    References:
        .. [DivRank] Mei, Q., Guo, J., & Radev, D. (2010, July). Divrank: the interplay
           of prestige and diversity in information networks. In Proceedings of the
           16th ACM SIGKDD international conference on Knowledge discovery and data
           mining (pp. 1009-1018). ACM. http://clair.si.umich.edu/~radev/papers/SIGKDD2010.pdf
    """
    # check function arguments
    if len(graph) == 0:
        LOGGER.warning("``graph`` is empty!")
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
        key=operator.itemgetter(1),
        reverse=True,
    )

    # replace node number by node value
    divranks = {nodes_list[result[0]]: result[1] for result in results}

    return divranks
