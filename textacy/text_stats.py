"""
Calculations for common "readability" statistics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from math import sqrt

from spacy.tokens import Doc as SpacyDoc

from textacy import Doc as TextacyDoc
from textacy import data
from textacy import extract


class TextStats(object):

    def __init__(self, doc):
        if isinstance(doc, SpacyDoc):
            lang = doc.vocab.lang
            self.n_sents = sum(1 for _ in doc.sents)
        elif isinstance(doc, TextacyDoc):
            lang = doc.lang
            self.n_sents = doc.n_sents
        else:
            raise ValueError('``doc`` must be a ``textacy.Doc`` or ``spacy.Doc``')
        hyphenator = data.load_hyphenator(lang=lang)
        words = tuple(extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False))
        syllables_per_word = tuple(len(hyphenator.positions(word.lower_)) + 1 for word in words)
        chars_per_word = tuple(len(word) for word in words)
        # compute basic stats needed for most other stats
        self.n_words = len(words)
        self.n_unique_words = len({word.lower for word in words})
        self.n_long_words = sum(1 for cpw in chars_per_word if cpw >= 7)
        self.n_chars = sum(chars_per_word)
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
    def gulpease_index(self):
        return gulpease_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def wiener_sachtextformel(self):
        return wiener_sachtextformel(
            self.n_words, self.n_polysyllable_words, self.n_monosyllable_words,
            self.n_long_words, self.n_sents,
            variant=1)

    @property
    def basic_stats(self):
        return {'n_sents': self.n_sents,
                'n_words': self.n_words,
                'n_chars': self.n_chars,
                'avg_sent_length': self.n_words / self.n_sents,
                'avg_word_length': self.n_chars / self.n_words,
                'n_unique_words': self.n_unique_words,
                'n_monosyllable_words': self.n_monosyllable_words,
                'n_polysyllable_words': self.n_polysyllable_words}

    @property
    def readability_stats(self):
        return {'flesch_kincaid_grade_level': self.flesch_kincaid_grade_level,
                'flesch_readability_ease': self.flesch_readability_ease,
                'smog_index': self.smog_index,
                'gunning_fog_index': self.gunning_fog_index,
                'coleman_liau_index': self.coleman_liau_index,
                'automated_readability_index': self.automated_readability_index,
                'gulpease_index': self.gulpease_index}


def readability_stats(doc):
    """
    Get calculated values for a variety of statistics related to the "readability"
    of a text: Flesch-Kincaid Grade Level, Flesch Reading Ease, SMOG Index,
    Gunning-Fog Index, Coleman-Liau Index, and Automated Readability Index.

    Also includes constituent values needed to compute the stats, e.g. word count.

    Args:
        doc (:class:`textacy.Doc <textacy.document.Doc>`)

    Returns:
        dict: mapping of readability statistic name (str) to value (int or float)

    Raises:
        NotImplementedError: if ``doc`` is not English language. sorry.
    """
    if doc.lang != 'en':
        raise NotImplementedError('non-English NLP is not ready yet, sorry')

    n_sents = doc.n_sents

    words = list(extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False))
    n_words = len(words)
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
        logging.warning('SMOG score may be unreliable for n_sents < 30')
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


def wiener_sachtextformel(n_words, n_polysyllable_words, n_monosyllable_words,
                          n_long_words, n_sents,
                          variant=1):
    """https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel"""
    ms = 100 * n_polysyllable_words / n_words
    sl = n_words / n_sents
    iw = 100 * n_long_words / n_words
    es = 100 * n_monosyllable_words / n_words
    if variant == 1:
        return 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
    elif variant == 2:
        return 0.2007 * ms + 0.1682 * sl + 0.1373 * iw - 2.779
    elif variant == 3:
        return 0.2963 * ms + 0.1905 * sl - 1.1144
    elif variant == 4:
        return 0.2744 * ms + 0.2656 * sl - 1.693
    else:
        raise ValueError('``variant`` value invalid; must be 1, 2, 3, or 4')


def gulpease_index(n_chars, n_words, n_sents):
    """https://it.wikipedia.org/wiki/Indice_Gulpease"""
    return 89 + (300 * n_sents / n_words) - (10 * n_chars / n_words)
