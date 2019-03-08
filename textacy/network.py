"""
Semantic Networks
-----------------

Represent documents as semantic networks, where nodes are individual terms or
whole sentences and edges are weighted by the strength of their co-occurrence or
similarity, respectively.
"""
import collections
import itertools
import logging

import networkx as nx
from cytoolz import itertoolz
from spacy.tokens.span import Span as SpacySpan
from spacy.tokens.token import Token as SpacyToken

from . import compat
from . import extract
from . import vsm

LOGGER = logging.getLogger(__name__)


def terms_to_semantic_network(
    terms, normalize="lemma", window_width=10, edge_weighting="cooc_freq"
):
    """
    Transform an ordered list of non-overlapping terms into a semantic network,
    where each term is represented by a node with weighted edges linking it to
    other terms that co-occur within ``window_width`` terms of itself.

    Args:
        terms (List[str] or List[``spacy.Token``])
        normalize (str or Callable): If 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if false-y, use the form of terms as they appear
            in ``terms``; if a callable, must accept a ``spacy.Token`` and return
            a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`.

            .. note:: This is applied to the elements of ``terms`` *only* if
               it's a list of ``spacy.Token``.

        window_width (int): Size of sliding window over ``terms`` that determines
            which are said to co-occur. If 2, only immediately adjacent terms
            have edges in the returned network.
        edge_weighting ({'cooc_freq', 'binary'}): If 'cooc_freq', the nodes for
            all co-occurring terms are connected by edges with weight equal to
            the number of times they co-occurred within a sliding window;
            if 'binary', all such edges have weight = 1.

    Returns:
        ``networkx.Graph``: Nodes in this network correspond to individual terms;
        those that co-occur are connected by edges with weights determined
        by ``edge_weighting``.

    Notes:
        - Be sure to filter out stopwords, punctuation, certain parts of speech, etc.
          from the terms list before passing it to this function
        - Multi-word terms, such as named entities and compound nouns, must be merged
          into single strings or spacy.Tokens beforehand
        - If terms are already strings, be sure to have normalized them so that
          like terms are counted together; for example, by applying
          :func:`textacy.spacier.utils.get_normalized_text()`
    """
    if window_width < 2:
        raise ValueError(
            "`window_width` = {} is invalid; value must be >= 2".format(window_width)
        )
    if not terms:
        LOGGER.warning("input `terms` is empty, so output graph is also empty")
        return nx.Graph()

    # if len(terms) < window_width, cytoolz throws a StopIteration error
    # which we don't want
    if len(terms) < window_width:
        LOGGER.info(
            "`terms` has fewer items (%s) than the specified `window_width` (%s); "
            "setting window width to %s",
            len(terms),
            window_width,
            len(terms),
        )
        window_width = len(terms)

    if isinstance(terms[0], compat.unicode_):
        windows = itertoolz.sliding_window(window_width, terms)
    elif isinstance(terms[0], SpacyToken):
        if normalize == "lemma":
            windows = (
                (tok.lemma_ for tok in window)
                for window in itertoolz.sliding_window(window_width, terms)
            )
        elif normalize == "lower":
            windows = (
                (tok.lower_ for tok in window)
                for window in itertoolz.sliding_window(window_width, terms)
            )
        elif not normalize:
            windows = (
                (tok.text for tok in window)
                for window in itertoolz.sliding_window(window_width, terms)
            )
        else:
            windows = (
                (normalize(tok) for tok in window)
                for window in itertoolz.sliding_window(window_width, terms)
            )
    else:
        raise TypeError(
            "items in `terms` must be strings or spacy tokens, not {}".format(
                type(terms[0])
            )
        )

    graph = nx.Graph()

    if edge_weighting == "cooc_freq":
        cooc_mat = collections.defaultdict(lambda: collections.defaultdict(int))
        for window in windows:
            for w1, w2 in itertools.combinations(sorted(window), 2):
                cooc_mat[w1][w2] += 1
        graph.add_edges_from(
            (w1, w2, {"weight": weight})
            for w1, w2s in cooc_mat.items()
            for w2, weight in w2s.items()
        )
    elif edge_weighting == "binary":
        graph.add_edges_from(
            w1_w2 for window in windows for w1_w2 in itertools.combinations(window, 2)
        )

    return graph


def sents_to_semantic_network(sents, normalize="lemma", edge_weighting="cosine"):
    """
    Transform a list of sentences into a semantic network, where each sentence is
    represented by a node with edges linking it to other sentences weighted by
    the (cosine or jaccard) similarity of their constituent words.

    Args:
        sents (List[str] or List[``spacy.Span``])
        normalize (str or Callable): If 'lemma', lemmatize words in sents;
            if 'lower', lowercase word in sents; if false-y, use the form of words
            as they appear in sents; if a callable, must accept a ``spacy.Token``
            and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`.

            .. note:: This is applied to the elements of ``sents`` *only* if
               it's a list of ``spacy.Span``.

        edge_weighting ({'cosine', 'jaccard'}): Similarity metric to use for
            weighting edges between sentences. If 'cosine', use the cosine
            similarity between sentences represented as tf-idf word vectors;
            if 'jaccard', use the set intersection divided by the set union of
            all words in a given sentence pair.

    Returns:
        ``networkx.Graph``: Nodes are the integer indexes of the sentences
        in ``sents``, *not* the actual text of the sentences! Edges connect
        every node, with weights determined by ``edge_weighting``.

    Notes:
        - If passing sentences as strings, be sure to filter out stopwords, punctuation,
          certain parts of speech, etc. beforehand
        - Consider normalizing the strings so that like terms are counted together
          (see :func:`textacy.spacier.utils.get_normalized_text()`)
    """
    if isinstance(sents[0], compat.unicode_):
        pass
    elif isinstance(sents[0], SpacySpan):
        if normalize == "lemma":
            sents = (
                (
                    tok.lemma_
                    for tok in extract.words(
                        sent, filter_stops=True, filter_punct=True, filter_nums=False
                    )
                )
                for sent in sents
            )
        elif normalize == "lower":
            sents = (
                (
                    tok.lower_
                    for tok in extract.words(
                        sent, filter_stops=True, filter_punct=True, filter_nums=False
                    )
                )
                for sent in sents
            )
        elif not normalize:
            sents = (
                (
                    tok.text
                    for tok in extract.words(
                        sent, filter_stops=True, filter_punct=True, filter_nums=False
                    )
                )
                for sent in sents
            )
        else:
            sents = (
                (
                    normalize(tok)
                    for tok in extract.words(
                        sent, filter_stops=True, filter_punct=True, filter_nums=False
                    )
                )
                for sent in sents
            )
    else:
        raise TypeError(
            "items in `sents` must be strings or spacy tokens, not {}".format(
                type(sents[0])
            )
        )

    if edge_weighting == "cosine":
        term_sent_matrix = vsm.Vectorizer(
            tf_type="linear", apply_idf=True, idf_type="smooth"
        ).fit_transform(sents)
    elif edge_weighting == "jaccard":
        term_sent_matrix = vsm.Vectorizer(
            tf_type="binary", apply_idf=False
        ).fit_transform(sents)
    weights = (term_sent_matrix * term_sent_matrix.T).A.tolist()
    n_sents = len(weights)

    graph = nx.Graph()
    graph.add_edges_from(
        (i, j, {"weight": weights[i][j]})
        for i in compat.range_(n_sents)
        for j in compat.range_(i + 1, n_sents)
    )

    return graph
