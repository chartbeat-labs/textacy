"""
Functions for computing basic text statistics.
"""
import logging
import math
from typing import Iterable, Tuple, Union

import spacy.pipeline
from cytoolz import itertoolz
from spacy.tokens import Doc, Token

from .. import extract
from . import api as _api


LOGGER = logging.getLogger(__name__)

_SENTENCIZER = spacy.pipeline.Sentencizer()


def n_words(doc_or_words: Union[Doc, Iterable[Token]]) -> int:
    """
    Compute the number of words in a document.

    Args:
        doc_or_words: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all are included as-is.
    """
    words = _get_words(doc_or_words)
    return itertoolz.count(words)


def n_unique_words(doc_or_words: Union[Doc, Iterable[Token]]) -> int:
    """
    Compute the number of *unique* words in a document.

    Args:
        doc_or_words: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all are included as-is.
    """
    words = _get_words(doc_or_words)
    # NOTE: this stdlib solution is slower than itertoolz for docs with ~250+ words
    # so let's take a small hit on short docs for the sake of big wins on long docs
    # return len({word.lower for word in words})
    return itertoolz.count(itertoolz.unique(word.lower for word in words))


def n_chars_per_word(doc_or_words: Union[Doc, Iterable[Token]]) -> Tuple[int, ...]:
    """
    Compute the number of characters for each word in a document.

    Args:
        doc_or_words: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all are included as-is.
    """
    words = _get_words(doc_or_words)
    return tuple(len(word) for word in words)


def n_chars(n_chars_per_word: Tuple[int, ...]) -> int:
    """
    Compute the total number of characters in a document.

    Args:
        n_chars_per_word: Number of characters per word in a given document,
            as computed by :func:`n_chars_per_word()`.
    """
    return sum(n_chars_per_word)


def n_long_words(n_chars_per_word: Tuple[int, ...], min_n_chars: int = 7) -> int:
    """
    Compute the number of long words in a document.

    Args:
        n_chars_per_word: Number of characters per word in a given document,
            as computed by :func:`n_chars_per_word()`.
        min_n_chars: Minimum number of characters required for a word to be
            considered "long".
    """
    return itertoolz.count(nc for nc in n_chars_per_word if nc >= min_n_chars)


def n_syllables_per_word(
    doc_or_words: Union[Doc, Iterable[Token]], lang: str,
) -> Tuple[int, ...]:
    """
    Compute the number of syllables for each word in a document.

    Args:
        doc_or_words: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all are included as-is.

    Note:
        Identifying syllables is _tricky_; this method relies on hyphenation, which is
        more straightforward but doesn't always give the correct number of syllables.
        While all hyphenation points fall on syllable divisions, not all syllable
        divisions are valid hyphenation points.
    """
    hyphenator = _api.load_hyphenator(lang=lang)
    words = _get_words(doc_or_words)
    return tuple(len(hyphenator.positions(word.lower_)) + 1 for word in words)


def n_syllables(n_syllables_per_word: Tuple[int, ...]) -> int:
    """
    Compute the total number of syllables in a document.

    Args:
        n_syllables_per_word: Number of syllables per word in a given document,
            as computed by :func:`n_syllables_per_word()`.
    """
    return sum(n_syllables_per_word)


def n_monosyllable_words(n_syllables_per_word: Tuple[int, ...]) -> int:
    """
    Compute the number of monosyllobic words in a document.

    Args:
        n_syllables_per_word: Number of syllables per word in a given document,
            as computed by :func:`n_syllables_per_word()`.
    """
    return itertoolz.count(ns for ns in n_syllables_per_word if ns == 1)


def n_polysyllable_words(
    n_syllables_per_word: Tuple[int, ...], min_n_syllables: int = 3,
) -> int:
    """
    Compute the number of polysyllobic words in a document.

    Args:
        n_syllables_per_word: Number of syllables per word in a given document,
            as computed by :func:`n_syllables_per_word()`.
        min_n_syllables: Minimum number of syllables required for a word to be
            considered "polysyllobic".
    """
    return itertoolz.count(ns for ns in n_syllables_per_word if ns >= min_n_syllables)


def _get_words(doc_or_words: Union[Doc, Iterable[Token]]) -> Iterable[Token]:
    if isinstance(doc_or_words, Doc):
        return extract.words(
            doc_or_words, filter_punct=True, filter_stops=False, filter_nums=False,
        )
    else:
        return doc_or_words


def n_sents(doc: Doc) -> int:
    """
    Compute the number of sentences in a document.

    Warning:
        If ``doc`` has not been segmented into sentences, it will be modified in-place
        using spaCy's rule-based ``Sentencizer`` pipeline component before counting.
    """
    if not doc.is_sentenced:
        LOGGER.warning(
            "`doc` has not been segmented into sentences; applying spaCy's rule-based, "
            "`Sentencizer` pipeline component to `doc` before counting..."
        )
        doc = _SENTENCIZER(doc)
    return itertoolz.count(doc.sents)


def entropy(doc_or_words: Union[Doc, Iterable[Token]]) -> float:
    """
    Compute the entropy of words in a document.

    Args:
        doc_or_words: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all are included as-is.
    """
    words = _get_words(doc_or_words)
    word_counts = itertoolz.frequencies(word.text for word in words)
    n_words = sum(word_counts.values())
    probs = (count / n_words for count in word_counts.values())
    return -sum(prob * math.log2(prob) for prob in probs)
