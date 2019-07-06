# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import math
import operator

from cytoolz import itertoolz

from . import utils as ke_utils
from .. import compat, utils


def yake(doc, ngrams=(1, 2, 3), window_size=2, match_thresh=0.75, topn=10):
    """
    Extract key terms from a document using the YAKE algorithm.

    Args:
        doc (:class:`spacy.tokens.Doc`): spaCy document.
            Must be sentence-segmented; optionally POS-tagged.
        ngrams (int or Set[int]): n of which n-grams to consider as keyterm candidates.
            For example, `(1, 2, 3)`` includes all unigrams, bigrams, and trigrams,
            while ``2`` includes bigrams only.
        window_size (int): Number of words to the right and left of a given word
            to use as context when computing the "relatedness to context"
            component of its score.
        match_thresh (float)
        topn (int or float): Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(candidates) * topn))``

    Returns:
        List[Tuple[str, float]]: Sorted list of top ``topn`` key terms and
        their corresponding scores.

    Reference:
        Campos, Mangaravite, Pasquali, Jorge, Nunes, and Jatowt. (2018).
        A Text Feature Based Automatic Keyword Extraction Method for Single Documents.
        Advances in Information Retrieval. ECIR 2018.
        Lecture Notes in Computer Science, vol 10772, pp. 684-691.
    """
    # validate / transform args
    ngrams = utils.to_collection(ngrams, int, tuple)
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                "topn={} is invalid; "
                "must be an int or a float between 0.0 and 1.0".format(topn)
            )

    stop_words = set()
    seen_candidates = set()
    # compute key values on a per-word basis
    word_occ_vals = _get_per_word_occurrence_values(doc, stop_words, window_size)
    # doc doesn't have any words...
    if not word_occ_vals:
        return []

    word_freqs = {w_id: len(vals["is_uc"]) for w_id, vals in word_occ_vals.items()}
    word_scores = _compute_word_scores(doc, word_occ_vals, word_freqs, stop_words)
    # compute scores for candidate terms based on scores of constituent words
    term_scores = {}
    # do single-word candidates separately; it's faster and simpler
    if 1 in ngrams:
        candidates = _get_unigram_candidates(doc)
        _score_unigram_candidates(
            candidates,
            word_freqs, word_scores, term_scores,
            stop_words, seen_candidates,
        )
    # now compute combined scores for higher-n ngram and candidates
    candidates = _get_ngram_candidates(doc, tuple(n for n in ngrams if n > 1))
    ngram_freqs = itertoolz.frequencies(
        " ".join(word.lower_ for word in ngram)
        for ngram in candidates)
    _score_ngram_candidates(
        candidates,
        ngram_freqs, word_scores, term_scores,
        seen_candidates,
    )
    # build up a list of key terms in order of increasing score
    if isinstance(topn, float):
        topn = int(round(len(seen_candidates) * topn))
    sorted_term_scores = sorted(
        term_scores.items(),
        key=operator.itemgetter(1),
        reverse=False,
    )
    return ke_utils.get_topn_terms(
        sorted_term_scores, topn, match_threshold=match_thresh)


def _get_per_word_occurrence_values(doc, stop_words, window_size):
    """
    Get base values for each individual occurrence of a word, to be aggregated
    and combined into a per-word score.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        stop_words (Set[str])
        window_size (int)

    Returns:
        Dict[int, Dict[str, list]]
    """
    word_occ_vals = collections.defaultdict(lambda: collections.defaultdict(list))

    def _is_upper_cased(tok):
        return tok.is_upper or (tok.is_title and not tok.is_sent_start)

    padding = [None] * window_size
    for sent_idx, sent in enumerate(doc.sents):
        sent_padded = itertoolz.concatv(padding, sent, padding)
        for window in itertoolz.sliding_window(1 + (2 * window_size), sent_padded):
            lwords, word, rwords = window[:window_size], window[window_size], window[window_size + 1:]
            w_id = word.lower
            if word.is_stop:
                stop_words.add(w_id)
            word_occ_vals[w_id]["is_uc"].append(_is_upper_cased(word))
            word_occ_vals[w_id]["sent_idx"].append(sent_idx)
            word_occ_vals[w_id]["l_context"].extend(
                w.lower for w in lwords
                if not (w is None or w.is_punct or w.is_space)
            )
            word_occ_vals[w_id]["r_context"].extend(
                w.lower for w in rwords
                if not (w is None or w.is_punct or w.is_space)
            )
    return word_occ_vals


def _compute_word_scores(doc, word_occ_vals, word_freqs, stop_words):
    """
    Aggregate values from per-word occurrence values, compute per-word weights
    of several components, then combine components into per-word scores.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        word_occ_vals (Dict[int, Dict[str, list]])
        word_freqs (Dict[int, int])
        stop_words (Set[str])

    Returns:
        Dict[int, float]
    """
    word_weights = collections.defaultdict(dict)
    # compute summary stats for word frequencies
    freqs_nsw = [freq for w_id, freq in word_freqs.items() if w_id not in stop_words]
    freq_max = max(word_freqs.values())
    freq_baseline = compat.mean_(freqs_nsw) + compat.stdev_(freqs_nsw)
    n_sents = itertoolz.count(doc.sents)
    for w_id, vals in word_occ_vals.items():
        freq = word_freqs[w_id]
        word_weights[w_id]["case"] = sum(vals["is_uc"]) / compat.log2_(1 + freq)
        word_weights[w_id]["pos"] = compat.log2_(compat.log2_(3 + compat.median_(vals["sent_idx"])))
        word_weights[w_id]["freq"] = freq / freq_baseline
        word_weights[w_id]["disp"] = len(set(vals["sent_idx"])) / n_sents
        n_unique_lc = len(set(vals["l_context"]))
        n_unique_rc = len(set(vals["r_context"]))
        try:
            wl = n_unique_lc / len(vals["l_context"])
        except ZeroDivisionError:
            wl = 0.0
        try:
            wr = n_unique_rc / len(vals["r_context"])
        except ZeroDivisionError:
            wr = 0.0
        pl = n_unique_lc / freq_max
        pr = n_unique_rc / freq_max
        word_weights[w_id]["rel"] = 1.0 + (wl + wr) * (freq / freq_max) + pl + pr

    # combine individual weights into per-word scores
    word_scores = {
        w_id: (wts["rel"] * wts["pos"]) / (wts["case"] + (wts["freq"] / wts["rel"]) + (wts["disp"] / wts["rel"]))
        for w_id, wts in word_weights.items()
    }
    return word_scores


def _get_unigram_candidates(doc):
    """
    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        List[:class:`spacy.tokens.Token`]
    """
    candidates = (
        word for word in doc
        if not (word.is_punct or word.is_space)
    )
    if doc.is_tagged:
        include_pos = {"NOUN", "PROPN", "ADJ"}
        candidates = (
            word for word in candidates
            if word.pos_ in include_pos
        )
    return candidates


def _get_ngram_candidates(doc, ngrams):
    """
    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (Tuple[int])

    Returns:
        List[Tuple[:class:`spacy.tokens.Token`]]
    """
    ngrams = itertoolz.concatv(*(itertoolz.sliding_window(n, doc) for n in ngrams))
    ngrams = (
        ngram
        for ngram in ngrams
        if not (ngram[0].is_stop or ngram[-1].is_stop)
        and not any(word.is_punct or word.is_space for word in ngram)
    )
    if doc.is_tagged:
        include_pos = {"NOUN", "PROPN", "ADJ"}
        ngrams = [
            ngram
            for ngram in ngrams
            if all(word.pos_ in include_pos for word in ngram)
        ]
    else:
        ngrams = list(ngrams)
    return ngrams


def _score_unigram_candidates(
    candidates,
    word_freqs,
    word_scores,
    term_scores,
    stop_words,
    seen_candidates,
):
    """
    Args:
        candidates (List[:class:`spacy.tokens.Token`])
        word_freqs (Dict[int, float])
        word_scores (Dict[int, float])
        term_scores (Dict[str, float])
        stop_words (Set[str])
        seen_candidates (Set[str])
    """
    for word in candidates:
        w_id = word.lower
        if w_id in stop_words or w_id in seen_candidates:
            continue
        else:
            seen_candidates.add(w_id)
        # NOTE: here i've modified the YAKE algorithm to put less emphasis on term freq
        # term_scores[word.lower_] = word_scores[w_id] / (word_freqs[w_id] * (1 + word_scores[w_id]))
        term_scores[word.lower_] = word_scores[w_id] / (compat.log2_(1 + word_freqs[w_id]) * (1 + word_scores[w_id]))


def _score_ngram_candidates(
    candidates,
    ngram_freqs, word_scores, term_scores,
    seen_candidates,
):
    """
    Args:
        candidates (List[Tuple[:class:`spacy.tokens.Token`]])
        ngram_freqs (Dict[str, int])
        word_scores (Dict[int, float])
        term_scores (Dict[str, float])
        seen_candidates (Set[str])
    """
    for ngram in candidates:
        ngtxt = " ".join(word.lower_ for word in ngram)
        if ngtxt in seen_candidates:
            continue
        else:
            seen_candidates.add(ngtxt)
        ngram_word_scores = [word_scores[word.lower] for word in ngram]
        # multiply individual word scores together in the numerator
        numerator = compat.reduce_(operator.mul, ngram_word_scores, 1.0)
        # NOTE: here i've modified the YAKE algorithm to put less emphasis on term freq
        # denominator = ngram_freqs[ngtxt] * (1.0 + sum(ngram_word_scores))
        denominator = compat.log2_(1 + ngram_freqs[ngtxt]) * (1.0 + sum(ngram_word_scores))
        term_scores[ngtxt] = numerator / denominator
