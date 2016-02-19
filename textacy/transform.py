"""
Functions to transform documents and corpora into other representations,
including term-document matrices, semantic networks, and ... WIP.
"""
import itertools
import networkx as nx
import numpy as np

from collections import defaultdict
from cytoolz import itertoolz
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import binarize as binarize_mat
from sklearn.preprocessing import normalize as normalize_mat
from spacy import attrs
from spacy.tokens.span import Span as spacy_span
from spacy.tokens.token import Token as spacy_token

from textacy import extract, text_stats
from textacy.compat import str
from textacy.spacy_utils import normalized_str

# TODO: bag-of-words? bag-of-concepts? gensim-compatible corpus and dictionary?


def terms_to_semantic_network(terms,
                              window_width=10,
                              edge_weighting='cooc_freq'):
    """
    Convert an ordered list of non-overlapping terms into a semantic network,
    where each terms is represented by a node with edges linking it to other terms
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
        :class:`networkx.Graph()`

    Notes:
        - Be sure to filter out stopwords, punctuation, certain parts of speech, etc.
          from the terms list before passing it to this function
        - Multi-word terms, such as named entities and compound nouns, must be merged
          into single strings or spacy.Tokens beforehand
        - If terms are already strings, be sure to normalize so that like terms
          are counted together (see :func:`normalized_str() <textacy.spacy_utils.normalized_str>`)
    """
    if window_width < 2:
        raise ValueError('Window width must be >= 2.')

    if isinstance(terms[0], str):
        windows = itertoolz.sliding_window(window_width, terms)
    elif isinstance(terms[0], spacy_token):
        windows = ((normalized_str(tok) for tok in window)
                   for window in itertoolz.sliding_window(window_width, terms))
    else:
        msg = 'Input terms must be strings or spacy Tokens, not {}.'.format(type(terms[0]))
        raise TypeError(msg)

    graph = nx.Graph()

    if edge_weighting == 'cooc_freq':
        cooc_mat = defaultdict(lambda: defaultdict(int))
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
        :class:`networkx.Graph()`: nodes are the integer indexes of the sentences
            in the input ``sents`` list, *not* the actual text of the sentences!

    Notes:
        * If passing sentences as strings, be sure to filter out stopwords, punctuation,
          certain parts of speech, etc. beforehand
        * Consider normalizing the strings so that like terms are counted together
          (see :func:`normalized_str() <textacy.spacy_utils.normalized_str>`)
    """
    n_sents = len(sents)
    if isinstance(sents[0], str):
        pass
    elif isinstance(sents[0], spacy_span):
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


def corpus_to_term_doc_matrix(corpus, weighting='tf',
                              normalize=True, binarize=False, smooth_idf=True,
                              min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None,
                              ngram_range=(1, 1), include_nes=False,
                              include_nps=False, include_kts=False):
    """
    Transform a collection of spacy docs (``corpus``) into a sparse CSR matrix, where
    each row i corresponds to a doc, each column j corresponds to a unique term.

    Args:
        corpus (:class:`TextCorpus <textacy.texts.TextCorpus>`)
        weighting (str {'tf', 'tfidf'}, optional): if 'tf', matrix values (i, j)
            correspond to the number of occurrences of term j in doc i; if 'tfidf',
            term frequencies (tf) are multiplied by their corresponding inverse
            document frequencies (idf)
        normalize (bool, optional): if True, normalize term frequencies by the
            L1 norms of the vectors
        binarize (bool, optional): if True, set all term frequencies greater than
            0 equal to 1
        smooth_idf (bool, optional): if True, add 1 to all document frequencies,
            equivalent to adding a single document to the corpus containing every
            unique term
        min_df (float or int, optional): if float, value is the fractional proportion
            of the total number of documents, which must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is less than ``min_df``
        max_df (float or int, optional): if float, value is the fractional proportion
            of the total number of documents, which must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is greater than ``max_df``
        min_ic (float, optional): filter terms whose information content is less
            than `min_ic`; value must be in [0.0, 1.0]
        max_n_terms (int, optional): only include terms whose document frequency
            is within the top ``max_n_terms``
        ngram_range (tuple(int, int), optional): range of ngrams to include as
            terms; default is unigrams only
        include_nes (bool, optional): if True, include named entities as terms
            (columns) in the matrix
        include_nps (bool, optional): if True, include noun phrases as terms
            (columns) in the matrix
        include_kts (bool, optional): if True, include SGRank key terms as terms
            (columns) in the matrix

    Returns:
        tuple(:class:`scipy.sparse.csr_matrix`, dict): 2-tuple of a weighted
            ``term_doc_matrix`` (an N X M matrix, where N is the # of docs, M is
            the # of unique terms, and value (n, m) is the weight of term m
            in doc n) and an ``id_to_term`` mapping (dict with unique integer
            term identifiers as keys and the corresponding normalized strings
            as values)
    """
    id_to_term = defaultdict()
    id_to_term.default_factory = id_to_term.__len__

    rows = []
    cols = []
    data = []
    for doc in corpus:
        term_counts = doc.term_counts(ngram_range=ngram_range,
                                      include_nes=include_nes,
                                      include_nps=include_nps,
                                      include_kts=include_kts)
        row = doc.corpus_index
        rows.extend(itertools.repeat(row, times=len(term_counts)))
        cols.extend(id_to_term[key] for key in term_counts.keys())
        data.extend(term_counts.values())

    id_to_term = {val: corpus.spacy_stringstore[key]
                  for key, val in id_to_term.items()}
    term_doc_matrix = sparse.coo_matrix((data, (rows, cols)),
                                        dtype=np.float64).tocsr()

    # filter terms by document frequency?
    if max_df != 1.0 or min_df != 1 or max_n_terms is not None:
        term_doc_matrix, id_to_term = text_stats.filter_terms_by_df(
            term_doc_matrix, id_to_term,
            max_df=max_df, min_df=min_df, max_n_terms=max_n_terms)
    if min_ic != 0.0:
        term_doc_matrix, id_to_term = text_stats.filter_terms_by_ic(
            term_doc_matrix, id_to_term,
            min_ic=min_ic, max_n_terms=max_n_terms)

    if normalize is True:
        term_doc_matrix = normalize_mat(term_doc_matrix, norm='l1', axis=1, copy=False)
    elif binarize is True:
        term_doc_matrix = binarize_mat(term_doc_matrix, threshold=0.0, copy=False)

    if weighting == 'tfidf':
        dfs = text_stats.get_doc_freqs(term_doc_matrix, normalized=False)
        if smooth_idf is True:
            n_docs = term_doc_matrix.shape[0] + 1
            dfs += 1
        idfs = np.log(n_docs / dfs) + 1.0
        term_doc_matrix = term_doc_matrix.multiply(idfs)

    return term_doc_matrix, id_to_term


def corpus_to_cscmatrix(corpus, lemmatize=False):
    """
    Transform a list of spacy docs (`corpus`) into a sparse CSC matrix, where
    each row i corresponds to a unique term, each column j corresponds to a doc,
    and values (i, j) correspond to the number of occurrences of term i in doc j.

    Args:
        corpus (list(``spacy.Doc``))
        lemmatize (bool, optional)

    Returns:
        :class:`scipy.sparse.csc_matrix`
    """
    num_nnz = 0
    data = []
    indices = []
    indptr = [0]

    for doc in corpus:
        if lemmatize is False:
            term_freqs = list(doc.count_by(attrs.ORTH).items())
        else:
            term_freqs = list(doc.count_by(attrs.LEMMA).items())
        indices.extend(term for term, _ in term_freqs)
        data.extend(freq for _, freq in term_freqs)
        num_nnz += len(term_freqs)
        indptr.append(num_nnz)

    num_terms = max(indices) + 1 if indices else 0
    num_docs = len(indptr) - 1
    data = np.asarray(data, dtype=np.int32)
    indices = np.asarray(indices)

    return sparse.csc_matrix((data, indices, indptr),
                             shape=(num_terms, num_docs),
                             dtype=np.int32)
