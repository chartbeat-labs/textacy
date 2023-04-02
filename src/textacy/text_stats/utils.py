"""
:mod:`textacy.text_stats.utils`: Utility functions for computing text statistics,
called under the hood of many stats functions -- and not typically accessed by users.
"""
import functools
import logging
from typing import Iterable

import pyphen
from cachetools import cached
from cachetools.keys import hashkey
from spacy.tokens import Token
from toolz import itertoolz

from .. import cache, types


LOGGER = logging.getLogger(__name__)


def get_words(doc_or_tokens: types.DocOrTokens) -> Iterable[Token]:
    """
    Get all non-punct, non-space tokens -- "words" as we commonly understand them --
    from input ``Doc`` or ``Iterable[Token]`` object.
    """
    words = (tok for tok in doc_or_tokens if not (tok.is_punct or tok.is_space))
    yield from words


def compute_n_words_and_types(words: Iterable[Token]) -> tuple[int, int]:
    """
    Compute the number of words and the number of unique words (aka types).

    Args:
        words: Sequence of non-punct, non-space tokens -- "words" -- as output, say,
            by :func:`get_words()`.

    Returns:
        (n_words, n_types)
    """
    word_counts = itertoolz.frequencies(word.lower for word in words)
    return (sum(word_counts.values()), len(word_counts))


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "hyphenator"))
def load_hyphenator(lang: str):
    """
    Load an object that hyphenates words at valid points, as used in LaTex typesetting.

    Args:
        lang: Standard 2-letter language abbreviation. To get a list of valid values::

            >>> import pyphen; pyphen.LANGUAGES

    Returns:
        :class:`pyphen.Pyphen()`
    """
    LOGGER.debug("loading '%s' language hyphenator", lang)
    return pyphen.Pyphen(lang=lang)
