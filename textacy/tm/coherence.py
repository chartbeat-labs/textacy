"""
References:
    .. Röder, Michael, Andreas Both, and Alexander Hinneburg. "Exploring the
        space of topic coherence measures." Proceedings of the eighth ACM
        international conference on Web search and data mining. ACM, 2015.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
from math import log

from cytoolz import dicttoolz, itertoolz
import numpy as np
import scipy.sparse as sp
from spacy.strings import StringStore

# WIP WIP WIP

def get_tm_coherence(model, id2term, n_terms=10,
                     segmentation='one-set',
                     probabilities='sliding-window',
                     direct_confirmation='npmi',
                     indirect_confirmation='cosine',
                     aggregation='mean'):
    """
    """
    all_sims = []
    for i, topic_words in enumerate(textacy.topic_modeling.get_top_topic_terms(
            model, id2term, n_topics=-1, n_terms=10, weights=False)):

        topic_lexemes = tuple(spacy_vocab[topic_word] for topic_word in topic_words)
        topic_lexemes = tuple(tl for tl in topic_lexemes if tl.has_vector)
        for term_subset1, term_subset2 in term_subset_pairs(topic_lexemes, segmentation=segmentation):
            cv1 = np.array(tuple(pow(sum(term1.similarity(term2) for term1 in term_subset1), 1)
                                 for term2 in topic_lexemes))
            cv2 = np.array(tuple(pow(sum(term1.similarity(term2) for term1 in term_subset2), 1)
                                 for term2 in topic_lexemes))
            sim = vector_similarity(cv1, cv2, metric='cosine')
            all_sims.append(sim)

    return aggregate_measures(all_sims, method=aggregation)


def get_topic_coherence(topic_words, segmentation='one-set', aggregation='mean'):
    pass


def compute_term_probabilities(spacy_docs, top_terms, lemmatize=True,
                               method='sliding-window', window_width=50):
    """
    Args:
        spacy_docs (iterable(``spacy.Doc``))
        top_terms (set(str))
        lemmatize (bool, optional)
        method ({'sliding-window', 'sentence', 'document'}, optional)
        window_width (int, optional)

    Returns:
        dict
    """
    stringstore = StringStore()
    top_terms = {stringstore[term] for term in top_terms}

    if method == 'sliding-window':
        if lemmatize is True:
            subdocs = (frozenset(stringstore[tok.lemma_] for tok in window
                                 if tok.lemma_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs
                       for window in itertoolz.sliding_window(min(window_width, len(spacy_doc)), spacy_doc))
        else:
            subdocs = (frozenset(stringstore[tok.orth_] for tok in window
                                 if tok.orth_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs
                       for window in itertoolz.sliding_window(min(window_width, len(spacy_doc)), spacy_doc))
    elif method == 'sentence':
        if lemmatize is True:
            subdocs = (frozenset(stringstore[tok.lemma_] for tok in sent
                                 if tok.lemma_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs
                       for sent in spacy_doc.sents)
        else:
            subdocs = (frozenset(stringstore[tok.orth_] for tok in sent
                                 if tok.orth_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs
                       for sent in spacy_doc.sents)
    elif method == 'document':
        if lemmatize is True:
            subdocs = (frozenset(stringstore[tok.lemma_] for tok in spacy_doc
                                 if tok.lemma_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs)
        else:
            subdocs = (frozenset(stringstore[tok.orth_] for tok in sent
                                 if tok.orth_ and not tok.is_stop
                                 and not tok.is_punct if not tok.is_space).intersection(top_terms)
                       for spacy_doc in spacy_docs)
    else:
        raise ValueError()

    # confusing bit of code... it iterates over all sub-documents,
    # concatenating all single and term-pair combinations into one iterable per sub-doc,
    # as well as adding 1 to an iterable once per sub-doc
    # then it splits these two streams, summing the 1s to get the total number of sub-docs
    # and concatenating all sub-docs into a single iterable used to initialize a counter in one go
    # it's slightly faster than just using a for loop over all subdocs... :shrug:
    ones, sds = zip(*((1, itertoolz.concatv(subdoc, itertools.combinations(sorted(subdoc), 2)))
                    for subdoc in subdocs))
    n_subdocs = sum(ones)
    term_probs = collections.Counter(itertoolz.concat(sds))

    if n_subdocs > 1:
        term_probs = dicttoolz.valmap(lambda x: x / n_subdocs, term_probs,
                                      factory=collections.Counter)

    return term_probs


def get_term_subset_pairs(terms, segmentation='one-set'):
    """
    Segment a word set ``terms`` into a set of pairs of subsets, where subset pairs
    consist of two parts: the first part is the subset for which support by the
    second part of the pair will be determined.

    Args:
        terms (set(str))
        segmentation ({'one-one', 'one-pre', 'one-suc', 'one-all', 'one-set'})

    Returns:
        list(tuple(tuple(str), tuple(str)))

    Raises:
        ValueError: if segmentation is not an allowed value
    """
    if segmentation == 'one-one':
        return [((term1,), (term2,))
                for term1, term2 in itertools.permutations(terms, r=2)]
    elif segmentation == 'one-pre':
        return [((term1,), (term2,))
                for i, term1 in enumerate(terms)
                for term2 in terms[:i]]
    elif segmentation == 'one-suc':
        return [((term1,), (term2,))
                for i, term1 in enumerate(terms)
                for term2 in terms[i+1:]]
    elif segmentation == 'one-all':
        return [((term1,), tuple(term2 for term2 in terms if term2 != term1))
                for term1 in terms]
    elif segmentation == 'one-set':
        return [((term1,), tuple(terms))
                for term1 in terms]
    else:
        msg = 'segmentation "{}" invalid; must be in {}'.format(
            segmentation, {'one-one', 'one-pre', 'one-suc', 'one-all', 'one-set'})
        raise ValueError(msg)


def pmi(term1, term2, term_probs, epsilon=1e-12):
    if term1 > term2:
        term1, term2 = term2, term1
    return log((term_probs[term1+'|'+term2] + epsilon) / (term_probs[term1] * term_probs[term2]))


def normalized_pmi(term1, term2, term_probs, epsilon=1e-12):
    if term1 > term2:
        term1, term2 = term2, term1
    joint_prob = term_probs[term1+'|'+term2] + epsilon
    pmi = log(joint_prob / (term_probs[term1] * term_probs[term2]))
    norm_pmi = pmi / (-1*log(joint_prob))
    return norm_pmi


def context_vector(term_subset1, term_subset2, term_counts, n_singles, n_pairs,
                   epsilon=1e-12, gamma=1):
    return np.array(tuple(pow(sum(normalized_pmi(term1, term2, term_probs, epsilon=epsilon)
                                  for term1 in term_subset1),
                              gamma)
                          for term2 in term_subset2))


def vector_similarity(vec1, vec2, metric='cosine'):
    if metric == 'cosine':
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    elif metric == 'dice':
        return np.minimum(cv1, cv2).sum() / (cv1 + cv2).sum()
    elif metric == 'jaccard':
        return np.minimum(cv1, cv2).sum() / np.maximum(cv1, cv2).sum()


def aggregate_measures(measures, method='mean'):
    if method == 'mean':
        return np.mean(measures)
    elif method == 'median':
        return np.median(measures)
    else:
        msg = 'measures "{}" invalid; must be in {}'.format(measures, {'mean', 'median'})
        raise ValueError(msg)
