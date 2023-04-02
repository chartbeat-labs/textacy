from __future__ import annotations

import collections
import functools
import math
import operator
import statistics
from typing import Collection, Iterable, Optional

from cytoolz import itertoolz
from spacy.tokens import Doc, Token

from ... import errors, utils
from .. import utils as ext_utils


def yake(
    doc: Doc,
    *,
    normalize: Optional[str] = "lemma",
    ngrams: int | Collection[int] = (1, 2, 3),
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
    window_size: int = 2,
    topn: int | float = 10,
) -> list[tuple[str, float]]:
    """
    Extract key terms from a document using the YAKE algorithm.

    Args:
        doc: spaCy ``Doc`` from which to extract keyterms.
            Must be sentence-segmented; optionally POS-tagged.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms;
            if "norm", use the norm of the terms (as set in a language's tokenizer exceptions);
            if None, use the form of terms as they appeared in ``doc``.

            .. note:: Unlike the other keyterm extraction functions, this one
               doesn't accept a callable for ``normalize``.

        ngrams: n of which n-grams to consider as keyterm candidates.
            For example, `(1, 2, 3)`` includes all unigrams, bigrams, and trigrams,
            while ``2`` includes bigrams only.
        include_pos: One or more POS tags with which to filter for good candidate keyterms.
            If None, include tokens of all POS tags
            (which also allows keyterm extraction from docs without POS-tagging.)
        window_size: Number of words to the right and left of a given word
            to use as context when computing the "relatedness to context"
            component of its score. Note that the resulting sliding window's
            full width is ``1 + (2 * window_size)``.
        topn: Number of top-ranked terms to return as key terms.
            If an integer, represents the absolute number; if a float, value
            must be in the interval (0.0, 1.0], which is converted to an int by
            ``int(round(len(candidates) * topn))``

    Returns:
        Sorted list of top ``topn`` key terms and their corresponding YAKE scores.

    References:
        Campos, Mangaravite, Pasquali, Jorge, Nunes, and Jatowt. (2018).
        A Text Feature Based Automatic Keyword Extraction Method for Single Documents.
        Advances in Information Retrieval. ECIR 2018.
        Lecture Notes in Computer Science, vol 10772, pp. 684-691.
    """
    # validate / transform args
    ngrams: tuple[int, ...] = utils.to_tuple(ngrams)
    include_pos: Optional[set[str]] = utils.to_set(include_pos) if include_pos else None
    if isinstance(topn, float):
        if not 0.0 < topn <= 1.0:
            raise ValueError(
                f"topn = {topn} is invalid; "
                "must be an int, or a float between 0.0 and 1.0"
            )

    # bail out on empty docs
    if not doc:
        return []

    stop_words: set[str] = set()
    seen_candidates: set[str] = set()
    # compute key values on a per-word basis
    word_occ_vals = _get_per_word_occurrence_values(
        doc, normalize, stop_words, window_size
    )
    # doc doesn't have any words...
    if not word_occ_vals:
        return []

    word_freqs = {w_id: len(vals["is_uc"]) for w_id, vals in word_occ_vals.items()}
    word_scores = _compute_word_scores(doc, word_occ_vals, word_freqs, stop_words)
    # compute scores for candidate terms based on scores of constituent words
    term_scores: dict[str, float] = {}
    # do single-word candidates separately; it's faster and simpler
    if 1 in ngrams:
        candidates = _get_unigram_candidates(doc, include_pos)
        _score_unigram_candidates(
            candidates,
            word_freqs,
            word_scores,
            term_scores,
            stop_words,
            seen_candidates,
            normalize,
        )
    # now compute combined scores for higher-n ngram and candidates
    candidates = list(
        ext_utils.get_ngram_candidates(
            doc,
            [n for n in ngrams if n > 1],
            include_pos=include_pos,
        )
    )
    attr_name = _get_attr_name(normalize, True)
    ngram_freqs = itertoolz.frequencies(
        " ".join(getattr(word, attr_name) for word in ngram) for ngram in candidates
    )
    _score_ngram_candidates(
        candidates,
        ngram_freqs,
        word_scores,
        term_scores,
        seen_candidates,
        normalize,
    )
    # build up a list of key terms in order of increasing score
    if isinstance(topn, float):
        topn = int(round(len(seen_candidates) * topn))
    sorted_term_scores = sorted(
        term_scores.items(),
        key=operator.itemgetter(1),
        reverse=False,
    )
    return ext_utils.get_filtered_topn_terms(
        sorted_term_scores, topn, match_threshold=0.8
    )


def _get_attr_name(normalize: Optional[str], as_strings: bool) -> str:
    if normalize is None:
        attr_name = "orth"
    elif normalize in ("lemma", "lower", "norm"):
        attr_name = normalize
    else:
        raise ValueError(
            errors.value_invalid_msg(
                "normalize", normalize, {"lemma", "lower", "norm", None}
            )
        )
    if as_strings is True:
        attr_name = attr_name + "_"
    return attr_name


def _get_per_word_occurrence_values(
    doc: Doc,
    normalize: Optional[str],
    stop_words: set[str],
    window_size: int,
) -> dict[int, dict[str, list]]:
    """
    Get base values for each individual occurrence of a word, to be aggregated
    and combined into a per-word score.
    """
    word_occ_vals: collections.defaultdict = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    def _is_upper_cased(tok):
        return tok.is_upper or (tok.is_title and not tok.is_sent_start)

    attr_name = _get_attr_name(normalize, False)
    padding = [None] * window_size
    for sent_idx, sent in enumerate(doc.sents):
        sent_padded = itertoolz.concatv(padding, sent, padding)
        for window in itertoolz.sliding_window(1 + (2 * window_size), sent_padded):
            lwords, word, rwords = (
                window[:window_size],
                window[window_size],
                window[window_size + 1 :],
            )
            w_id = getattr(word, attr_name)
            if word.is_stop:
                stop_words.add(w_id)
            word_occ_vals[w_id]["is_uc"].append(_is_upper_cased(word))
            word_occ_vals[w_id]["sent_idx"].append(sent_idx)
            word_occ_vals[w_id]["l_context"].extend(
                getattr(w, attr_name)
                for w in lwords
                if not (w is None or w.is_punct or w.is_space)
            )
            word_occ_vals[w_id]["r_context"].extend(
                getattr(w, attr_name)
                for w in rwords
                if not (w is None or w.is_punct or w.is_space)
            )
    return word_occ_vals


def _compute_word_scores(
    doc: Doc,
    word_occ_vals: dict[int, dict[str, list]],
    word_freqs: dict[int, int],
    stop_words: set[str],
) -> dict[int, float]:
    """
    Aggregate values from per-word occurrence values, compute per-word weights
    of several components, then combine components into per-word scores.
    """
    word_weights: collections.defaultdict = collections.defaultdict(dict)
    # compute summary stats for word frequencies
    freqs_nsw = [freq for w_id, freq in word_freqs.items() if w_id not in stop_words]
    freq_max = max(word_freqs.values())
    freq_baseline = statistics.mean(freqs_nsw) + statistics.stdev(freqs_nsw)
    n_sents = itertoolz.count(doc.sents)
    for w_id, vals in word_occ_vals.items():
        freq = word_freqs[w_id]
        word_weights[w_id]["case"] = sum(vals["is_uc"]) / math.log2(1 + freq)
        word_weights[w_id]["pos"] = math.log2(
            math.log2(3 + statistics.mean(vals["sent_idx"]))
        )
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
        w_id: (wts["rel"] * wts["pos"])
        / (wts["case"] + (wts["freq"] / wts["rel"]) + (wts["disp"] / wts["rel"]))
        for w_id, wts in word_weights.items()
    }
    return word_scores


def _get_unigram_candidates(
    doc: Doc, include_pos: Optional[set[str]]
) -> Iterable[Token]:
    candidates = (
        word for word in doc if not (word.is_stop or word.is_punct or word.is_space)
    )
    if include_pos:
        candidates = (word for word in candidates if word.pos_ in include_pos)
    return candidates


def _score_unigram_candidates(
    candidates: Iterable[Token],
    word_freqs: dict[int, int],
    word_scores: dict[int, float],
    term_scores: dict[str, float],
    stop_words: set[str],
    seen_candidates: set[str],
    normalize: Optional[str],
):
    attr_name = _get_attr_name(normalize, False)
    attr_name_str = _get_attr_name(normalize, True)
    for word in candidates:
        w_id = getattr(word, attr_name)
        if w_id in stop_words or w_id in seen_candidates:
            continue
        else:
            seen_candidates.add(w_id)
        # NOTE: here i've modified the YAKE algorithm to put less emphasis on term freq
        # term_scores[word.lower_] = word_scores[w_id] / (word_freqs[w_id] * (1 + word_scores[w_id]))
        term_scores[getattr(word, attr_name_str)] = word_scores[w_id] / (
            math.log2(1 + word_freqs[w_id]) * (1 + word_scores[w_id])
        )


def _score_ngram_candidates(
    candidates: list[tuple[Token, ...]],
    ngram_freqs: dict[str, int],
    word_scores: dict[int, float],
    term_scores: dict[str, float],
    seen_candidates: set[str],
    normalize: Optional[str],
):
    attr_name = _get_attr_name(normalize, False)
    attr_name_str = _get_attr_name(normalize, True)
    for ngram in candidates:
        ngtxt = " ".join(getattr(word, attr_name_str) for word in ngram)
        if ngtxt in seen_candidates:
            continue
        else:
            seen_candidates.add(ngtxt)
        ngram_word_scores = [word_scores[getattr(word, attr_name)] for word in ngram]
        # multiply individual word scores together in the numerator
        numerator = functools.reduce(operator.mul, ngram_word_scores, 1.0)
        # NOTE: here i've modified the YAKE algorithm to put less emphasis on term freq
        # denominator = ngram_freqs[ngtxt] * (1.0 + sum(ngram_word_scores))
        denominator = math.log2(1 + ngram_freqs[ngtxt]) * (1.0 + sum(ngram_word_scores))
        term_scores[ngtxt] = numerator / denominator
