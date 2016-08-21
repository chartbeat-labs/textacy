"""
Represent documents as semantic networks, where nodes are individual terms or
whole sentences.
"""
import collections
import itertools
import logging

from cytoolz import itertoolz
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokens.span import Span as SpacySpan
from spacy.tokens.token import Token as SpacyToken

from textacy.compat import unicode_type
from textacy import extract
from textacy.spacy_utils import normalized_str


logger = logging.getLogger(__name__)


def terms_to_semantic_network(terms,
                              window_width=10,
                              edge_weighting='cooc_freq'):
    """
    Convert an ordered list of non-overlapping terms into a semantic network,
    where each term is represented by a node with edges linking it to other terms
    that co-occur within ``window_width`` terms of itself.

    Args:
        terms (list(str) or list(``spacy.Token``))
        window_width (int, optional): size of sliding window over `terms` that
            determines which are said to co-occur; if = 2, only adjacent terms
            will have edges in network
        edge_weighting (str {'cooc_freq', 'binary'}, optional): if 'binary',
            all co-occurring terms will have network edges with weight = 1;
            if 'cooc_freq', edges will have a weight equal to the number of times
            that the connected nodes co-occur in a sliding window

    Returns:
        :class:`networkx.Graph <networkx.Graph>`: nodes are terms, edges are for
            co-occurrences of terms

    Notes:
        - Be sure to filter out stopwords, punctuation, certain parts of speech, etc.
          from the terms list before passing it to this function
        - Multi-word terms, such as named entities and compound nouns, must be merged
          into single strings or spacy.Tokens beforehand
        - If terms are already strings, be sure to normalize so that like terms
          are counted together (see :func:`normalized_str() <textacy.spacy_utils.normalized_str>`)
    """
    if window_width < 2:
        raise ValueError('Window width must be >= 2')

    # if len(terms) < window_width, cytoolz throws a StopIteration error
    # which we don't want
    if len(terms) < window_width:
        logger.warning(
            'input terms list is smaller than window width ({} < {})'.format(
                len(terms), window_width))
        window_width = len(terms)

    if isinstance(terms[0], unicode_type):
        windows = itertoolz.sliding_window(window_width, terms)
    elif isinstance(terms[0], SpacyToken):
        windows = ((normalized_str(tok) for tok in window)
                   for window in itertoolz.sliding_window(window_width, terms))
    else:
        msg = 'Input terms must be strings or spacy Tokens, not {}.'.format(type(terms[0]))
        raise TypeError(msg)

    graph = nx.Graph()

    if edge_weighting == 'cooc_freq':
        cooc_mat = collections.defaultdict(lambda: collections.defaultdict(int))
        for window in windows:
            for w1, w2 in itertools.combinations(sorted(window), 2):
                cooc_mat[w1][w2] += 1
        graph.add_edges_from(
            (w1, w2, {'weight': cooc_mat[w1][w2]})
            for w1, w2s in cooc_mat.items() for w2 in w2s)

    elif edge_weighting == 'binary':
        graph.add_edges_from(
            w1_w2 for window in windows
            for w1_w2 in itertools.combinations(window, 2))

    return graph


def sents_to_semantic_network(sents,
                              edge_weighting='cosine'):
    """
    Convert a list of sentences into a semantic network, where each sentence is
    represented by a node with edges linking it to other sentences weighted by
    the (cosine or jaccard) similarity of their constituent words.

    Args:
        sents (list(str) or list(:class:`spacy.Span`))
        edge_weighting (str {'cosine', 'jaccard'}, optional): similarity metric
            to use for weighting edges between sentences; if 'cosine', use the
            cosine similarity between sentences represented as tf-idf word vectors;
            if 'jaccard', use the set intersection divided by the set union of
            all words in a given sentence pair

    Returns:
        :class:`networkx.Graph`: nodes are the integer indexes of the sentences
            in the input ``sents`` list, *not* the actual text of the sentences!

    Notes:
        * If passing sentences as strings, be sure to filter out stopwords, punctuation,
          certain parts of speech, etc. beforehand
        * Consider normalizing the strings so that like terms are counted together
          (see :func:`normalized_str() <textacy.spacy_utils.normalized_str>`)
    """
    n_sents = len(sents)
    if isinstance(sents[0], unicode_type):
        pass
    elif isinstance(sents[0], SpacySpan):
        sents = [' '.join(normalized_str(tok) for tok in
                          extract.words(sent, filter_stops=True, filter_punct=True, filter_nums=False))
                 for sent in sents]
    else:
        msg = 'Input sents must be strings or spacy Spans, not {}.'.format(type(sents[0]))
        raise TypeError(msg)

    if edge_weighting == 'cosine':
        term_sent_matrix = TfidfVectorizer().fit_transform(sents)
    elif edge_weighting == 'jaccard':
        term_sent_matrix = CountVectorizer(binary=True).fit_transform(sents)
    weights = (term_sent_matrix * term_sent_matrix.T).A.tolist()

    graph = nx.Graph()
    graph.add_edges_from(
        (i, j, {'weight': weights[i][j]})
        for i in range(n_sents) for j in range(i + 1, n_sents))

    return graph
