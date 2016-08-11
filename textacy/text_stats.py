"""
Calculations for common "readability" statistics. English only, for now.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt

from textacy import data
from textacy import extract


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
    return 11.8 * (n_syllables / n_words) + 0.39 * (n_words / n_sents) - 15.59


def flesch_readability_ease(n_syllables, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease"""
    return -84.6 * (n_syllables / n_words) - 1.015 * (n_words / n_sents) + 206.835


def smog_index(n_polysyllable_words, n_sents, verbose=False):
    """https://en.wikipedia.org/wiki/SMOG"""
    if verbose and n_sents < 30:
        print('**WARNING: SMOG score may be unreliable for n_sents < 30')
    return 1.0430 * sqrt(30 * (n_polysyllable_words / n_sents)) + 3.1291


def gunning_fog_index(n_words, n_polysyllable_words, n_sents):
    """https://en.wikipedia.org/wiki/Gunning_fog_index"""
    return 0.4 * ((n_words / n_sents) + 100 * (n_polysyllable_words / n_words))


def coleman_liau_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index"""
    return 5.879851 * (n_chars / n_words) - 29.587280 * (n_sents / n_words) - 15.800804


def automated_readability_index(n_chars, n_words, n_sents):
    """https://en.wikipedia.org/wiki/Automated_readability_index"""
    return 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
