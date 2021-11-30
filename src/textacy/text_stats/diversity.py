"""
Lexical Diversity Stats
-----------------------

:mod:`textacy.text_stats.diversity`: Low-level functions for computing various measures
of lexical diversity, typically accessed via :meth:`textacy.text_stats.TextStats.diversity()`.
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Iterable, Literal

from scipy.stats import hypergeom
from spacy.tokens import Token
from toolz import itertoolz

from .. import errors, types
from . import utils


LOGGER = logging.getLogger(__name__)


def ttr(
    doc_or_tokens: types.DocOrTokens,
    variant: Literal["standard", "root", "corrected"] = "standard",
) -> float:
    """
    Compute the Type-Token Ratio (TTR) of ``doc_or_tokens``,
    a direct ratio of the number of unique words (types) to all words (tokens).

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        variant: Particular variant of TTR.
            - "standard" => ``n_types / n_words``
            - "root" => ``n_types / sqrt(n_words)``
            - "corrected" => ``n_types / sqrt(2 * n_words)``

    Note:
        All variants of this measure are sensitive to document length,
        so values from texts with different lengths should not be compared.

    References:
        - Templin, M. (1957). Certain language skills in children. Minneapolis:
          University of Minnesota Press.
        - RTTR: Guiraud 1954, 1960
        - CTTR: 1964 Carrol
    """
    n_words, n_types = utils.compute_n_words_and_types(utils.get_words(doc_or_tokens))
    try:
        if variant == "standard":
            return n_types / n_words
        elif variant == "root":
            return n_types / math.sqrt(n_words)
        elif variant == "corrected":
            return n_types / math.sqrt(2 * n_words)
        else:
            raise ValueError(
                errors.value_invalid_msg(
                    "variant", variant, {"standard", "root", "corrected"}
                )
            )
    except ZeroDivisionError:
        return 0.0


def log_ttr(
    doc_or_tokens: types.DocOrTokens,
    variant: Literal["herdan", "summer", "dugast"] = "herdan",
) -> float:
    """
    Compute the logarithmic Type-Token Ratio (TTR) of ``doc_or_tokens``,
    a modification of TTR that uses log functions to better adapt for text length.

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        variant: Particular variant of log-TTR.
            - "herdan" => ``log(n_types) / log(n_words)``
            - "summer" => ``log(log(n_types)) / log(log(n_words))``
            - "dugast" => ``log(n_words) ** 2 / (log(n_words) - log(n_types))``

    Note:
        All variants of this measure are slightly sensitive to document length,
        so values from texts with different lengths should be compared with care.

        The popular Maas variant of log-TTR is simply the reciprocal of Dugast's:
        ``(log(n_words) - log(n_types)) / log(n_words) ** 2``. It isn't included
        as a variant because its interpretation differs: *lower* values
        indicate higher lexical diversity.

    References:
        - Herdan, G. (1964). Quantitative linguistics. London: Butterworths.
        - Somers, H. H. (1966). Statistical methods in literary analysis. In J. Leeds
          (Ed.), The computer and literary style (pp. 128-140). Kent, OH: Kent
          State University.
        - Dugast, D. (1978). Sur quoi se fonde la notion d’étendue théoretique du
          vocabulaire? Le Français Moderne, 46, 25-32.
    """
    n_words, n_types = utils.compute_n_words_and_types(utils.get_words(doc_or_tokens))
    try:
        if variant == "herdan":
            return math.log10(n_types) / math.log10(n_words)
        elif variant == "summer":
            return math.log10(math.log10(n_types)) / math.log10(math.log10(n_words))
        elif variant == "dugast":
            log10_n_words = math.log10(n_words)
            return (log10_n_words ** 2) / (log10_n_words - math.log10(n_types))
        else:
            raise ValueError(
                errors.value_invalid_msg(
                    "variant", variant, {"herdan", "summer", "dugast"}
                )
            )
    except ZeroDivisionError:
        return 0.0


def segmented_ttr(
    doc_or_tokens: types.DocOrTokens,
    segment_size: int = 50,
    variant: Literal["mean", "moving-avg"] = "mean",
) -> float:
    """
    Compute the Mean Segmental TTR (MS-TTR) or Moving Average TTR (MA-TTR)
    of ``doc_or_tokens``, in which the TTR of tumbling or rolling segments of words,
    respectively, each with length ``segment_size``, are computed and then averaged.

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        segment_size: Number of consecutive words to include in each segment.
        variant: Variant of segmented TTR to compute.
            - "mean" => MS-TTR
            - "moving-avg" => MA-TTR

    References:
        - Johnson, W. (1944). Studies in language behavior: I. A program of research.
          Psychological Monographs, 56, 1-15.
        - Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot:
          The moving-average type–token ratio (MATTR). Journal of quantitative
          linguistics, 17(2), 94-100.
    """
    words = tuple(utils.get_words(doc_or_tokens))
    if len(words) < segment_size:
        LOGGER.warning(
            "number of words in document (%s) must be greater than segment size (%s) "
            "to compute segmented-TTR; setting segment size equal to number of words, "
            "which effectively reduces this method to standard TTR",
            len(words),
            segment_size,
        )
        segment_size = len(words)
    if variant == "mean":
        # TODO: keep or drop shorter last segments?
        segments = itertoolz.partition(segment_size, words)
        # segments = itertoolz.partition_all(segment_size, words)
    elif variant == "moving-avg":
        segments = itertoolz.sliding_window(segment_size, words)
    else:
        raise ValueError(
            errors.value_invalid_msg("variant", variant, {"mean", "moving-avg"})
        )
    return statistics.mean(ttr(seg_words) for seg_words in segments)


def mtld(doc_or_tokens: types.DocOrTokens, min_ttr: float = 0.72) -> float:
    """
    Compute the Measure of Textual Lexical Diversity (MTLD) of ``doc_or_tokens``,
    the average length of the longest consecutive sequences of words that maintain a TTR
    of at least ``min_ttr``.

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        min_ttr: Minimum TTR for each segment in ``doc_or_tokens``. When an ongoing
            segment's TTR falls below this value, a new segment is started.
            Value should be in the range [0.66, 0.75].

    References:
        McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study
        of sophisticated approaches to lexical diversity assessment. Behavior research
        methods, 42(2), 381-392.
    """
    words = tuple(utils.get_words(doc_or_tokens))
    return statistics.mean(
        [_mtld_run(words, min_ttr), _mtld_run(reversed(words), min_ttr)]
    )


def _mtld_run(words: Iterable[Token], min_ttr: float) -> float:
    n_factors = 0.0
    n_factor_words = 0
    factor_types = set()
    for n, word in enumerate(words):
        n_factor_words += 1
        factor_types.add(word.lower)
        ttr = len(factor_types) / n_factor_words
        if ttr < min_ttr:
            n_factors += 1.0
            n_factor_words = 0
            factor_types.clear()
    n_words = n + 1
    # do we have a partial factor left over? don't throw it away!
    # instead, add a fractional factor value corresponding to how far TTR is from 1.0
    # compared to how far min_ttr is from 1.0
    if n_factor_words > 0:
        n_factors += (1.0 - ttr) / (1 - min_ttr)
    # to avoid ZeroDivisionError in case of a single, partial factor with TTR == 1.0
    # we add a small constant; but tbh this is a weird case
    return n_words / (n_factors + 1e-6)


def hdd(doc_or_tokens: types.DocOrTokens, sample_size: int = 42) -> float:
    """
    Compute the Hypergeometric Distribution Diversity (HD-D) of ``doc_or_tokens``,
    which calculates the mean contribution that each unique word (aka type) makes
    to the TTR of all possible combinations of random samples of words of a given size,
    then sums all contributions together.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        sample_size: Number of words randomly sampled without replacement when
            computing unique word appearance probabilities.
            Value should be in the range [35, 50].

    Note:
        The popular vocd-D index of lexical diversity is actually just an approximation
        of HD-D, and should not be used.

    References:
        - McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study
          of sophisticated approaches to lexical diversity assessment. Behavior research
          methods, 42(2), 381-392.
        - McCarthy, P. M., & Jarvis, S. (2007). A theoretical and empirical evaluation
          of vocd. Language Testing, 24, 459-488.
    """
    words = utils.get_words(doc_or_tokens)
    type_counts = itertoolz.frequencies(word.lower for word in words)
    n_words = sum(type_counts.values())
    if n_words < sample_size:
        LOGGER.warning(
            "number of words in document (%s) must be greater than sample size (%s) "
            "to compute HD-D; setting sample size equal to number of words, "
            "which effectively reduces this method to standard TTR",
            n_words,
            sample_size,
        )
        sample_size = n_words

    type_contributions = (
        (1.0 - hypergeom.pmf(0, n_words, type_count, sample_size)) / sample_size
        for type_count in type_counts.values()
    )
    return sum(type_contributions)
