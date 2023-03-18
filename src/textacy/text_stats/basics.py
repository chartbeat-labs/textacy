"""
Basic Stats
-----------

:mod:`textacy.text_stats.basics`: Low-level functions for computing basic text statistics,
typically accessed via :class:`textacy.text_stats.TextStats`.
"""
from __future__ import annotations

import functools
import logging
import math
from typing import Optional

import spacy.pipeline
from cytoolz import itertoolz
from spacy.tokens import Doc

from .. import types
from . import utils


LOGGER = logging.getLogger(__name__)

_SENTENCIZER = spacy.pipeline.Sentencizer()


@functools.lru_cache(maxsize=128)
def n_sents(doc: Doc) -> int:
    """
    Compute the number of sentences in a document.

    Args:
        doc

    Warning:
        If ``doc`` has not been segmented into sentences, it will be modified in-place
        using spaCy's rule-based ``Sentencizer`` pipeline component before counting.
    """
    if not doc.has_annotation("SENT_START"):
        LOGGER.warning(
            "`doc` has not been segmented into sentences; applying spaCy's rule-based, "
            "`Sentencizer` pipeline component to `doc` before counting..."
        )
        doc = _SENTENCIZER(doc)
    return itertoolz.count(doc.sents)


def n_words(doc_or_tokens: types.DocOrTokens) -> int:
    """
    Compute the number of words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
    """
    words = utils.get_words(doc_or_tokens)
    return itertoolz.count(words)


def n_unique_words(doc_or_tokens: types.DocOrTokens) -> int:
    """
    Compute the number of *unique* words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
    """
    words = utils.get_words(doc_or_tokens)
    # NOTE: this stdlib solution is slower than itertoolz for docs with ~250+ words
    # so let's take a small hit on short docs for the sake of big wins on long docs
    # return len({word.lower for word in words})
    return itertoolz.count(itertoolz.unique(word.lower for word in words))


@functools.lru_cache(maxsize=128)
def n_chars_per_word(doc_or_tokens: types.DocOrTokens) -> tuple[int, ...]:
    """
    Compute the number of characters for each word in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.

    Note:
        This function is cached, since other functions rely upon its outputs
        to compute theirs. As such, ``doc_or_tokens`` must be hashable -- for example,
        it may be a ``Doc`` or ``tuple[Token, ...]`` , but not a ``List[Token]`` .
    """
    words = utils.get_words(doc_or_tokens)
    return tuple(len(word) for word in words)


def n_chars(doc_or_tokens: types.DocOrTokens) -> int:
    """
    Compute the total number of characters in a document's words.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.

    See Also:
        :func:`n_chars_per_word()`
    """
    # docs are hashable, so we can leverage the lru cache as-is
    if isinstance(doc_or_tokens, Doc):
        ncpw = n_chars_per_word(doc_or_tokens)
    # otherwise, let's get an iterable of words but cast it to a hashable tuple
    # so we can leverage the lru cache on this and related calls in, say, n_long_words
    else:
        words = utils.get_words(doc_or_tokens)
        ncpw = n_chars_per_word(tuple(words))
    return sum(ncpw)


def n_long_words(doc_or_tokens: types.DocOrTokens, *, min_n_chars: int = 7) -> int:
    """
    Compute the number of long words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        min_n_chars: Minimum number of characters required for a word to be
            considered "long".
    """
    # docs are hashable, so we can leverage the lru cache as-is
    if isinstance(doc_or_tokens, Doc):
        ncpw = n_chars_per_word(doc_or_tokens)
    # otherwise, let's get an iterable of words but cast it to a hashable tuple
    # so we can leverage the lru cache on this and related calls in, say, n_long_words
    else:
        words = utils.get_words(doc_or_tokens)
        ncpw = n_chars_per_word(tuple(words))
    return itertoolz.count(nc for nc in ncpw if nc >= min_n_chars)


@functools.lru_cache(maxsize=128)
def n_syllables_per_word(
    doc_or_tokens: types.DocOrTokens, *, lang: Optional[str] = None
) -> tuple[int, ...]:
    """
    Compute the number of syllables for each word in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        lang: Standard 2-letter language code used to load hyphenator.
            If not specified and ``doc_or_tokens`` is a spaCy ``Doc`` ,
            the value will be gotten from ``Doc.lang_`` .

    Note:
        Identifying syllables is _tricky_; this method relies on hyphenation, which is
        more straightforward but doesn't always give the correct number of syllables.
        While all hyphenation points fall on syllable divisions, not all syllable
        divisions are valid hyphenation points.

        Also: This function is cached, since other functions rely upon its outputs
        to compute theirs. As such, ``doc_or_tokens`` must be hashable -- for example,
        it may be a ``Doc`` or ``tuple[Token, ...]`` , but not a ``List[Token]`` .
    """
    if lang is None:
        if isinstance(doc_or_tokens, Doc):
            lang = doc_or_tokens.lang_
        else:
            raise ValueError(
                "`lang` must be specified when computing n syllables per word "
                "from an iterable of tokens"
            )
    hyphenator = utils.load_hyphenator(lang=lang)
    words = utils.get_words(doc_or_tokens)
    return tuple(len(hyphenator.positions(word.lower_)) + 1 for word in words)


def n_syllables(doc_or_tokens: types.DocOrTokens, *, lang: Optional[str] = None) -> int:
    """
    Compute the total number of syllables in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        lang: Standard 2-letter language code used to load hyphenator.
            If not specified and ``doc_or_tokens`` is a spaCy ``Doc`` ,
            the value will be gotten from ``Doc.lang_`` .

    See Also:
        :func:`n_syllables_per_word()`
    """
    # docs are hashable, so we can leverage the lru cache as-is
    if isinstance(doc_or_tokens, Doc):
        nspw = n_syllables_per_word(doc_or_tokens, lang=lang)
    # otherwise, let's get an iterable of words but cast it to a hashable tuple
    # so we can leverage the lru cache on this and related calls in, say, n_long_words
    else:
        words = utils.get_words(doc_or_tokens)
        nspw = n_syllables_per_word(tuple(words), lang=lang)
    return sum(nspw)


def n_monosyllable_words(
    doc_or_tokens: types.DocOrTokens, *, lang: Optional[str] = None
) -> int:
    """
    Compute the number of monosyllobic words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        lang: Standard 2-letter language code used to load hyphenator.
            If not specified and ``doc_or_tokens`` is a spaCy ``Doc`` ,
            the value will be gotten from ``Doc.lang_`` .

    See Also:
        :func:`n_syllables_per_word()`
    """
    # docs are hashable, so we can leverage the lru cache as-is
    if isinstance(doc_or_tokens, Doc):
        nspw = n_syllables_per_word(doc_or_tokens, lang=lang)
    # otherwise, let's get an iterable of words but cast it to a hashable tuple
    # so we can leverage the lru cache on this and related calls in, say, n_long_words
    else:
        words = utils.get_words(doc_or_tokens)
        nspw = n_syllables_per_word(tuple(words), lang=lang)
    return itertoolz.count(ns for ns in nspw if ns == 1)


def n_polysyllable_words(
    doc_or_tokens: types.DocOrTokens,
    *,
    lang: Optional[str] = None,
    min_n_syllables: int = 3,
) -> int:
    """
    Compute the number of polysyllobic words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
        lang: Standard 2-letter language code used to load hyphenator.
            If not specified and ``doc_or_tokens`` is a spaCy ``Doc`` ,
            the value will be gotten from ``Doc.lang_`` .
        min_n_syllables: Minimum number of syllables required for a word to be
            considered "polysyllobic".

    See Also:
        :func:`n_syllables_per_word()`
    """
    # docs are hashable, so we can leverage the lru cache as-is
    if isinstance(doc_or_tokens, Doc):
        nspw = n_syllables_per_word(doc_or_tokens, lang=lang)
    # otherwise, let's get an iterable of words but cast it to a hashable tuple
    # so we can leverage the lru cache on this and related calls in, say, n_long_words
    else:
        words = utils.get_words(doc_or_tokens)
        nspw = n_syllables_per_word(tuple(words), lang=lang)
    return itertoolz.count(ns for ns in nspw if ns >= min_n_syllables)


def entropy(doc_or_tokens: types.DocOrTokens) -> float:
    """
    Compute the entropy of words in a document.

    Args:
        doc_or_tokens: If a spaCy ``Doc``, non-punctuation tokens (words) are extracted;
            if an iterable of spaCy ``Token`` s, all non-punct elements are used.
    """
    words = utils.get_words(doc_or_tokens)
    word_counts = itertoolz.frequencies(word.text for word in words)
    n_words = sum(word_counts.values())
    probs = (count / n_words for count in word_counts.values())
    return -sum(prob * math.log2(prob) for prob in probs)
