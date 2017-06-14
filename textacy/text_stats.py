"""
Calculations of basic counts and readability statistics for text documents.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from math import sqrt
import warnings

from spacy.tokens import Doc as SpacyDoc

import textacy
from textacy import data, extract

LOGGER = logging.getLogger(__name__)


class TextStats(object):
    """
    Compute a variety of basic counts and readability statistics for a given
    text document. For example::

        >>> text = list(textacy.datasets.CapitolWords().texts(limit=1))[0]
        >>> doc = textacy.Doc(text)
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
         'flesch_readability_ease': 50.707745098039254,
         'gulpease_index': 51.86764705882353,
         'gunning_fog_index': 16.12549019607843,
         'lix': 54.28431372549019,
         'smog_index': 14.554592549557764,
         'wiener_sachtextformel': 8.266410784313727}

    Args:
        doc (:class:`textacy.Doc` or :class:`SpacyDoc`): A text document processed
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
        flesch_kincaid_grade_level (float): https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_grade_level
        flesch_readability_ease (float): https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
        smog_index (float): https://en.wikipedia.org/wiki/SMOG
        gunning_fog_index (float): https://en.wikipedia.org/wiki/Gunning_fog_index
        coleman_liau_index (float): https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        automated_readability_index (float): https://en.wikipedia.org/wiki/Automated_readability_index
        lix (float): https://en.wikipedia.org/wiki/LIX
        gulpease_index (float): https://it.wikipedia.org/wiki/Indice_Gulpease
        wiener_sachtextformel (float): https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel
            Note: This always returns variant #1.
        basic_counts (Dict[str, int]): Mapping of basic count names to values,
            where basic counts are the attributes listed above between ``n_sents``
            and ``n_polysyllable_words``.
        readability_stats (Dict[str, float]): Mapping of readability statistic
            names to values, where readability stats are the attributes listed
            above between ``flesch_kincaid_grade_level`` and ``wiener_sachtextformel``.

    Raises:
        ValueError: If ``doc`` is not a :class:`textacy.Doc` or :class:`SpacyDoc`.
    """

    def __init__(self, doc):
        if isinstance(doc, SpacyDoc):
            lang = doc.vocab.lang
            self.n_sents = sum(1 for _ in doc.sents)
        elif isinstance(doc, textacy.Doc):
            lang = doc.lang
            self.n_sents = doc.n_sents
        else:
            raise ValueError('``doc`` must be a ``textacy.Doc`` or ``spacy.Doc``')
        # get objs for basic count computations
        hyphenator = data.load_hyphenator(lang=lang)
        words = tuple(extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False))
        syllables_per_word = tuple(len(hyphenator.positions(word.lower_)) + 1 for word in words)
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
    def flesch_readability_ease(self):
        return flesch_readability_ease(self.n_syllables, self.n_words, self.n_sents)

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
            self.n_words, self.n_polysyllable_words, self.n_monosyllable_words,
            self.n_long_words, self.n_sents,
            variant=1)

    @property
    def basic_counts(self):
        return {'n_sents': self.n_sents,
                'n_words': self.n_words,
                'n_chars': self.n_chars,
                'n_syllables': self.n_syllables,
                'n_unique_words': self.n_unique_words,
                'n_long_words': self.n_long_words,
                'n_monosyllable_words': self.n_monosyllable_words,
                'n_polysyllable_words': self.n_polysyllable_words}

    @property
    def readability_stats(self):
        if self.n_words == 0:
            LOGGER.warning("readability stats can't be computed because doc has 0 words")
            return None
        return {'flesch_kincaid_grade_level': self.flesch_kincaid_grade_level,
                'flesch_readability_ease': self.flesch_readability_ease,
                'smog_index': self.smog_index,
                'gunning_fog_index': self.gunning_fog_index,
                'coleman_liau_index': self.coleman_liau_index,
                'automated_readability_index': self.automated_readability_index,
                'lix': self.lix,
                'gulpease_index': self.gulpease_index,
                'wiener_sachtextformel': self.wiener_sachtextformel}


def readability_stats(doc):
    """
    Get calculated values for a variety of statistics related to the "readability"
    of a text: Flesch-Kincaid Grade Level, Flesch Reading Ease, SMOG Index,
    Gunning-Fog Index, Coleman-Liau Index, and Automated Readability Index.

    Also includes constituent values needed to compute the stats, e.g. word count.

    **DEPRECATED**

    Args:
        doc (:class:`textacy.Doc <textacy.document.Doc>`)

    Returns:
        dict: mapping of readability statistic name (str) to value (int or float)

    Raises:
        NotImplementedError: if ``doc`` is not English language. sorry.
    """
    msg = '`readability_stats()` function is deprecated; use `TextStats` class instead'
    with warnings.catch_warnings():
        warnings.simplefilter('once', DeprecationWarning)
        warnings.warn(msg, DeprecationWarning)

    if doc.lang != 'en':
        raise NotImplementedError('non-English NLP is not ready yet, sorry')

    n_sents = doc.n_sents

    words = list(extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False))
    n_words = len(words)
    if n_words == 0:
        LOGGER.warning("readability stats can't be computed because doc has 0 words")
        return None
    n_unique_words = len({word.lower for word in words})
    n_chars = sum(len(word) for word in words)

    hyphenator = data.load_hyphenator(lang='en')
    syllables_per_word = [len(hyphenator.positions(word.lower_)) + 1 for word in words]
    n_syllables = sum(syllables_per_word)
    n_polysyllable_words = sum(1 for n in syllables_per_word if n >= 3)

    return {'n_sents': n_sents,
            'n_words': n_words,
            'n_unique_words': n_unique_words,
            'n_chars': n_chars,
            'n_syllables': n_syllables,
            'n_polysyllable_words': n_polysyllable_words,
            'flesch_kincaid_grade_level': flesch_kincaid_grade_level(n_syllables, n_words, n_sents),
            'flesch_readability_ease': flesch_readability_ease(n_syllables, n_words, n_sents),
            'smog_index': smog_index(n_polysyllable_words, n_sents),
            'gunning_fog_index': gunning_fog_index(n_words, n_polysyllable_words, n_sents),
            'coleman_liau_index': coleman_liau_index(n_chars, n_words, n_sents),
            'automated_readability_index': automated_readability_index(n_chars, n_words, n_sents)}


def flesch_kincaid_grade_level(n_syllables, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_grade_level"""
    return (11.8 * n_syllables / n_words) + (0.39 * n_words / n_sents) - 15.59


def flesch_readability_ease(n_syllables, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease"""
    return (-84.6 * n_syllables / n_words) - (1.015 * n_words / n_sents) + 206.835


def smog_index(n_polysyllable_words, n_sents):
    """https://en.wikipedia.org/wiki/SMOG"""
    if n_sents < 30:
        LOGGER.warning('SMOG score may be unreliable for n_sents < 30')
    return (1.0430 * sqrt(30 * n_polysyllable_words / n_sents)) + 3.1291


def gunning_fog_index(n_words, n_polysyllable_words, n_sents):
    """https://en.wikipedia.org/wiki/Gunning_fog_index"""
    return 0.4 * ((n_words / n_sents) + (100 * n_polysyllable_words / n_words))


def coleman_liau_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index"""
    return (5.879851 * n_chars / n_words) - (29.587280 * n_sents / n_words) - 15.800804


def automated_readability_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Automated_readability_index"""
    return (4.71 * n_chars / n_words) + (0.5 * n_words / n_sents) - 21.43


def lix(n_words, n_long_words, n_sents):
    """https://en.wikipedia.org/wiki/LIX"""
    return (n_words / n_sents) + (100 * n_long_words / n_words)


def wiener_sachtextformel(n_words, n_polysyllable_words, n_monosyllable_words,
                          n_long_words, n_sents,
                          variant=1):
    """https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel"""
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
        raise ValueError('``variant`` value invalid; must be 1, 2, 3, or 4')


def gulpease_index(n_chars, n_words, n_sents):
    """https://it.wikipedia.org/wiki/Indice_Gulpease"""
    return (300 * n_sents / n_words) - (10 * n_chars / n_words) + 89
