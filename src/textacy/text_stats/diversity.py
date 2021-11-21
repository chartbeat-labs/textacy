"""
Lexical Diversity
-----------------

:mod:`textacy.text_stats.diversity`: TODO
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable, Literal, Tuple

from scipy.stats import hypergeom
from spacy.tokens import Doc, Token
from toolz import itertoolz

from .. import errors
from . import basics


def _compute_n_words_and_types(words: Iterable[Token]) -> Tuple[int, int]:
    """
    Compute the number of all words and number of unique words (aka types).

    Returns:
        (n_words, n_types)
    """
    word_counts = itertoolz.frequencies(word.lower for word in words)
    return (sum(word_counts.values()), len(word_counts))


def ttr(
    doc_or_words: Doc | Iterable[Token],
    variant: Literal["standard", "root", "corrected"] = "standard",
) -> float:
    """
    Compute the Type-Token Ratio (TTR) of ``doc_or_words``,
    a direct ratio of the number of unique words (types) to all words (tokens).

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_words
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
    n_words, n_types = _compute_n_words_and_types(basics._get_words(doc_or_words))
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
    doc_or_words: Doc | Iterable[Token],
    variant: Literal["herdan", "summer", "dugast"] = "herdan",
) -> float:
    """
    Compute the logarithmic Type-Token Ratio (TTR) of ``doc_or_words``,
    a modification of TTR that uses log functions to better adapt for text length.

    Higher values indicate higher lexical diversity.

    Args:
        doc_or_words
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
    n_words, n_types = _compute_n_words_and_types(basics._get_words(doc_or_words))
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
    doc_or_words: Doc | Iterable[Token],
    segment_size: int = 50,
    variant: Literal["mean", "moving_avg"] = "mean",
) -> float:
    """
    Compute the Mean Segmental TTR (MS-TTR) or Moving Average TTR (MA-TTR)
    of ``doc_or_words``, in which the TTR of tumbling or rolling segments of words,
    respectively, each with length ``segment_size``, are computed and then averaged.

    Args:
        doc_or_words
        segment_size: Number of consecutive words to include in each segment.

    References:
        - Johnson, W. (1944). Studies in language behavior: I. A program of research.
          Psychological Monographs, 56, 1-15.
        - Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot:
          The moving-average type–token ratio (MATTR). Journal of quantitative
          linguistics, 17(2), 94-100.
    """
    words = list(basics._get_words(doc_or_words))
    if len(words) < segment_size:
        raise ValueError()

    if variant == "mean":
        # TODO: keep or drop shorter last segments?
        segments = itertoolz.partition(segment_size, words)
        # segments = itertoolz.partition_all(segment_size, words)
    elif variant == "moving_avg":
        segments = itertoolz.sliding_window(segment_size, words)
    else:
        raise ValueError(
            errors.value_invalid_msg("variant", variant, {"avg", "moving_avg"})
        )
    return statistics.mean(ttr(seg_words) for seg_words in segments)


def mtld(doc_or_words: Doc | Iterable[Token], min_ttr: float = 0.72) -> float:
    """
    Compute the Measure of Textual Lexical Diversity (MTLD) of ``doc_or_words``,
    the average length of the longest consecutive sequences of words that maintain a TTR
    of at least ``min_ttr``.

    Args:
        doc_or_words
        min_ttr: Minimum TTR for each segment in ``doc_or_words``. When an ongoing
            segment's TTR falls below this value, a new segment is started.
            Value should be in the range [0.66, 0.75].

    References:
        McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study
        of sophisticated approaches to lexical diversity assessment. Behavior research
        methods, 42(2), 381-392.
    """
    words = list(basics._get_words(doc_or_words))
    return statistics.mean(
        [_mtld_run(words, min_ttr), _mtld_run(reversed(words), min_ttr)]
    )


def _mtld_run(words: Iterable[Token], min_ttr: float) -> float:
    type_set = set()
    n_words = 0
    n_factor_words = 0
    n_factors = 0.0
    for word in words:
        n_words += 1
        n_factor_words += 1
        type_set.add(word.lower)
        ttr = len(type_set) / n_factor_words
        if ttr < min_ttr:
            n_factors += 1.0
            n_factor_words = 0
            type_set.clear()
    # TTR never falls below threshold, so we have a single, partial factor?
    if n_factors == 0:
        if ttr == 1:
            n_factors = 1.0
        else:
            n_factors += (1.0 - ttr) / (1 - min_ttr)
    # final, partial factor?
    elif n_factor_words > 0:
        n_factors += (1.0 - ttr) / (1 - min_ttr)

    return n_words / n_factors


def hdd(doc_or_words: Doc | Iterable[Token], sample_size: int = 42) -> float:
    """
    Compute the Hypergeometric Distribution Diversity (HD-D) of ``doc_or_words``,
    which calculates the mean contribution that each unique word (aka type) makes
    to the TTR of all possible combinations of random samples of words of a given size,
    then sums all contributions together.

    Args:
        doc_or_words
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
    words = basics._get_words(doc_or_words)
    type_counts = itertoolz.frequencies(word.lower for word in words)
    n_words = sum(type_counts.values())
    if n_words < sample_size:
        raise ValueError()

    type_contributions = (
        (1.0 - hypergeom.pmf(0, n_words, type_count, sample_size)) / sample_size
        for type_count in type_counts.values()
    )
    return sum(type_contributions)
