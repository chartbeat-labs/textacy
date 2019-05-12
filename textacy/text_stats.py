"""
Text Statistics
---------------

Compute a variety of basic counts and readability statistics for documents.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from math import sqrt

from . import cache
from . import extract
from . import utils

LOGGER = logging.getLogger(__name__)


class TextStats(object):
    """
    Compute a variety of basic counts and readability statistics for a given
    document. For example::

        >>> text = next(textacy.datasets.CapitolWords().texts(limit=1))
        >>> doc = textacy.make_spacy_doc(text)
        >>> ts = TextStats(doc)
        >>> ts.n_words
        136
        >>> ts.flesch_kincaid_grade_level
        11.817647058823532
        >>> ts.basic_counts
        {'n_chars': 685,
         'n_long_words': 43,
         'n_monosyllable_words': 90,
         'n_polysyllable_words': 24,
         'n_sents': 6,
         'n_syllables': 214,
         'n_unique_words': 80,
         'n_words': 136}
        >>> ts.readability_stats
        {'automated_readability_index': 13.626495098039214,
         'coleman_liau_index': 12.509300816176474,
         'flesch_kincaid_grade_level': 11.817647058823532,
         'flesch_reading_ease': 50.707745098039254,
         'gulpease_index': 51.86764705882353,
         'gunning_fog_index': 16.12549019607843,
         'lix': 54.28431372549019,
         'smog_index': 14.554592549557764,
         'wiener_sachtextformel': 8.266410784313727}

    Args:
        doc (:class:`spacy.tokens.Doc`): A text document processed
            by spacy. Need only be tokenized.

    Attributes:
        n_sents (int): Number of sentences in ``doc``.
        n_words (int): Number of words in ``doc``, including numbers + stop words
            but excluding punctuation.
        n_chars (int): Number of characters for all words in ``doc``.
        n_syllables (int): Number of syllables for all words in ``doc``.
        n_unique_words (int): Number of unique (lower-cased) words in ``doc``.
        n_long_words (int): Number of words in ``doc`` with 7 or more characters.
        n_monosyllable_words (int): Number of words in ``doc`` with 1 syllable only.
        n_polysyllable_words (int): Number of words in ``doc`` with 3 or more syllables.
            Note: Since this excludes words with exactly 2 syllables, it's likely
            that ``n_monosyllable_words + n_polysyllable_words != n_words``.
        flesch_kincaid_grade_level (float): see :func:`flesch_kincaid_grade_level()`
        flesch_reading_ease (float): see :func:`flesch_reading_ease()`
        smog_index (float): see :func:`smog_index()`
        gunning_fog_index (float): see :func:`gunning_fog_index()`
        coleman_liau_index (float): see :func:`coleman_liau_index()`
        automated_readability_index (float): see :func:`automated_readability_index()`
        lix (float): see :func:`lix()`
        gulpease_index (float): see :func:`gulpease_index()`
        wiener_sachtextformel (float): see :func:`wiener_sachtextformel()`
            Note: This always returns variant #1.
        basic_counts (Dict[str, int]): Mapping of basic count names to values,
            where basic counts are the attributes listed above between ``n_sents``
            and ``n_polysyllable_words``.
        readability_stats (Dict[str, float]): Mapping of readability statistic
            names to values, where readability stats are the attributes listed
            above between ``flesch_kincaid_grade_level`` and ``wiener_sachtextformel``.

    Raises:
        ValueError: If ``doc`` is not a :class:`spacy.tokens.Doc`.
    """

    def __init__(self, doc):
        self.lang = doc.vocab.lang
        self.n_sents = sum(1 for _ in doc.sents)
        # get objs for basic count computations
        hyphenator = cache.load_hyphenator(lang=self.lang)
        words = tuple(
            extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False)
        )
        syllables_per_word = tuple(
            len(hyphenator.positions(word.lower_)) + 1 for word in words
        )
        chars_per_word = tuple(len(word) for word in words)
        # compute basic counts needed for most readability stats
        self.n_words = len(words)
        self.n_unique_words = len({word.lower for word in words})
        self.n_chars = sum(chars_per_word)
        self.n_long_words = sum(1 for cpw in chars_per_word if cpw >= 7)
        self.n_syllables = sum(syllables_per_word)
        self.n_monosyllable_words = sum(1 for spw in syllables_per_word if spw == 1)
        self.n_polysyllable_words = sum(1 for spw in syllables_per_word if spw >= 3)

    @property
    def flesch_kincaid_grade_level(self):
        return flesch_kincaid_grade_level(self.n_syllables, self.n_words, self.n_sents)

    @property
    def flesch_reading_ease(self):
        return flesch_reading_ease(
            self.n_syllables, self.n_words, self.n_sents, lang=self.lang
        )

    @property
    def flesch_readability_ease(self):
        """For backwards compatibility. Deprecated."""
        return flesch_readability_ease(
            self.n_syllables, self.n_words, self.n_sents, lang=self.lang
        )

    @property
    def smog_index(self):
        return smog_index(self.n_polysyllable_words, self.n_sents)

    @property
    def gunning_fog_index(self):
        return gunning_fog_index(self.n_words, self.n_polysyllable_words, self.n_sents)

    @property
    def coleman_liau_index(self):
        return coleman_liau_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def automated_readability_index(self):
        return automated_readability_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def lix(self):
        return lix(self.n_words, self.n_long_words, self.n_sents)

    @property
    def gulpease_index(self):
        return gulpease_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def wiener_sachtextformel(self):
        return wiener_sachtextformel(
            self.n_words,
            self.n_polysyllable_words,
            self.n_monosyllable_words,
            self.n_long_words,
            self.n_sents,
            variant=1,
        )

    @property
    def basic_counts(self):
        return {
            "n_sents": self.n_sents,
            "n_words": self.n_words,
            "n_chars": self.n_chars,
            "n_syllables": self.n_syllables,
            "n_unique_words": self.n_unique_words,
            "n_long_words": self.n_long_words,
            "n_monosyllable_words": self.n_monosyllable_words,
            "n_polysyllable_words": self.n_polysyllable_words,
        }

    @property
    def readability_stats(self):
        if self.n_words == 0:
            LOGGER.warning(
                "readability stats can't be computed because doc has 0 words"
            )
            return None
        return {
            "flesch_kincaid_grade_level": self.flesch_kincaid_grade_level,
            "flesch_reading_ease": self.flesch_reading_ease,
            "smog_index": self.smog_index,
            "gunning_fog_index": self.gunning_fog_index,
            "coleman_liau_index": self.coleman_liau_index,
            "automated_readability_index": self.automated_readability_index,
            "lix": self.lix,
            "gulpease_index": self.gulpease_index,
            "wiener_sachtextformel": self.wiener_sachtextformel,
        }


def flesch_kincaid_grade_level(n_syllables, n_words, n_sents):
    """
    Readability score used widely in education, whose value estimates the U.S.
    grade level / number of years of education required to understand a text.
    Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_grade_level
    """
    return (11.8 * n_syllables / n_words) + (0.39 * n_words / n_sents) - 15.59


def flesch_reading_ease(n_syllables, n_words, n_sents, lang=None):
    """
    Readability score usually in the range [0, 100], related (inversely) to
    :func:`flesch_kincaid_grade_level()`. Higher value => easier text.

    Note:
        Constant weights in this formula are language-dependent;
        if ``lang`` is null, the English-language formulation is used.

    References:
        English: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
        German: https://de.wikipedia.org/wiki/Lesbarkeitsindex#Flesch-Reading-Ease
        Spanish: ?
        French: ?
        Italian: https://it.wikipedia.org/wiki/Formula_di_Flesch
        Dutch: ?
        Russian: https://ru.wikipedia.org/wiki/%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81_%D1%83%D0%B4%D0%BE%D0%B1%D0%BE%D1%87%D0%B8%D1%82%D0%B0%D0%B5%D0%BC%D0%BE%D1%81%D1%82%D0%B8
    """
    if lang is None or lang == "en":
        return 206.835 - (1.015 * n_words / n_sents) - (84.6 * n_syllables / n_words)
    elif lang == "de":
        return 180.0 - (n_words / n_sents) - (58.5 * n_syllables / n_words)
    elif lang == "es":
        return 206.84 - (1.02 * n_words / n_sents) - (60.0 * n_syllables / n_words)
    elif lang == "fr":
        return 207.0 - (1.015 * n_words / n_sents) - (73.6 * n_syllables / n_words)
    elif lang == "it":
        return 217.0 - (1.3 * n_words / n_sents) - (60.0 * n_syllables / n_words)
    elif lang == "nl":
        return 206.84 - (0.93 * n_words / n_sents) - (77.0 * n_syllables / n_words)
    elif lang == "ru":
        return 206.835 - (1.3 * n_words / n_sents) - (60.1 * n_syllables / n_words)
    else:
        langs = ["en", "de", "es", "fr", "it", "nl", "ru"]
        raise ValueError(
            "Flesch Reading Ease is only implemented for these languages: {}. "
            'Passing `lang=None` falls back to "en" (English)'.format(langs)
        )


def flesch_readability_ease(n_syllables, n_words, n_sents, lang=None):
    """
    Alias for :func:`flesch_reading_ease()`, for backwards compatibility.

    Deprecated!
    """
    utils.deprecated(
        "`flesch_readability_ease()` is an alias for `flesch_reading_ease()` "
        "for backwards compatibility; it will be removed in a future version.",
        action="once",
    )
    return flesch_reading_ease(n_syllables, n_words, n_sents, lang=lang)


def smog_index(n_polysyllable_words, n_sents):
    """
    Readability score commonly used in healthcare, whose value estimates the
    number of years of education required to understand a text, similar to
    :func:`flesch_kincaid_grade_level()` and intended as a substitute for
    :func:`gunning_fog_index()`. Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/SMOG
    """
    if n_sents < 30:
        LOGGER.warning("SMOG score may be unreliable for n_sents < 30")
    return (1.0430 * sqrt(30 * n_polysyllable_words / n_sents)) + 3.1291


def gunning_fog_index(n_words, n_polysyllable_words, n_sents):
    """
    Readability score whose value estimates the number of years of education
    required to understand a text, similar to :func:`flesch_kincaid_grade_level()`
    and :func:`smog_index()`. Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/Gunning_fog_index
    """
    return 0.4 * ((n_words / n_sents) + (100 * n_polysyllable_words / n_words))


def coleman_liau_index(n_chars, n_words, n_sents):
    """
    Readability score whose value estimates the number of years of education
    required to understand a text, similar to :func:`flesch_kincaid_grade_level()`
    and :func:`smog_index()`, but using characters instead of syllables.
    Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index"""
    return (5.879851 * n_chars / n_words) - (29.587280 * n_sents / n_words) - 15.800804


def automated_readability_index(n_chars, n_words, n_sents):
    """
    Readability score whose value estimates the U.S. grade level required to
    understand a text, most similarly to :func:`flesch_kincaid_grade_level()`,
    but using characters instead of syllables like :func:`coleman_liau_index()`.
    Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/Automated_readability_index
    """
    return (4.71 * n_chars / n_words) + (0.5 * n_words / n_sents) - 21.43


def lix(n_words, n_long_words, n_sents):
    """
    Readability score commonly used in Sweden, whose value estimates the
    difficulty of reading a foreign text. Higher value => more difficult text.

    References:
        https://en.wikipedia.org/wiki/LIX
    """
    return (n_words / n_sents) + (100 * n_long_words / n_words)


def wiener_sachtextformel(
    n_words,
    n_polysyllable_words,
    n_monosyllable_words,
    n_long_words,
    n_sents,
    variant=1,
):
    """
    Readability score for German-language texts, whose value estimates the grade
    level required to understand a text. Higher value => more difficult text.

    References:
        https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel
    """
    ms = 100 * n_polysyllable_words / n_words
    sl = n_words / n_sents
    iw = 100 * n_long_words / n_words
    es = 100 * n_monosyllable_words / n_words
    if variant == 1:
        return (0.1935 * ms) + (0.1672 * sl) + (0.1297 * iw) - (0.0327 * es) - 0.875
    elif variant == 2:
        return (0.2007 * ms) + (0.1682 * sl) + (0.1373 * iw) - 2.779
    elif variant == 3:
        return (0.2963 * ms) + (0.1905 * sl) - 1.1144
    elif variant == 4:
        return (0.2744 * ms) + (0.2656 * sl) - 1.693
    else:
        raise ValueError("``variant`` value invalid; must be 1, 2, 3, or 4")


def gulpease_index(n_chars, n_words, n_sents):
    """
    Readability score for Italian-language texts, whose value is in the range
    [0, 100] similar to :func:`flesch_reading_ease()`. Higher value =>
    easier text.

    References:
        https://it.wikipedia.org/wiki/Indice_Gulpease
    """
    return (300 * n_sents / n_words) - (10 * n_chars / n_words) + 89
