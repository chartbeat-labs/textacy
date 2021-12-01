"""
Readability Stats
-----------------

:mod:`textacy.text_stats.readability`: Low-level functions for computing various measures
of text "readability", typically accessed via :meth:`textacy.text_stats.TextStats.readability()`.
"""
import logging
import math
import statistics
from typing import Optional

from spacy.tokens import Doc

from .. import errors
from . import basics, utils


LOGGER = logging.getLogger(__name__)

_FRE_COEFS = {
    "de": {"base": 180.0, "asl": 1.0, "awl": 58.5},
    "en": {"base": 206.835, "asl": 1.015, "awl": 84.6},
    "es": {"base": 206.835, "asl": 1.02, "awl": 60.0},  # 0.6 x 100
    "fr": {"base": 207.0, "asl": 1.015, "awl": 73.6},
    "it": {"base": 217.0, "asl": 1.3, "awl": 60.0},  # 0.6 x 100
    "nl": {"base": 206.835, "asl": 0.93, "awl": 77.0},
    "pt": {"base": 248.835, "asl": 1.015, "awl": 84.6},
    "ru": {"base": 206.835, "asl": 1.3, "awl": 60.1},
    "tr": {"base": 198.825, "asl": 2.610, "awl": 40.175},
}


def automated_readability_index(doc: Doc) -> float:
    """
    Readability test for English-language texts, particularly for technical writing,
    whose value estimates the U.S. grade level required to understand a text.
    Similar to several other tests (e.g. :func:`flesch_kincaid_grade_level()`),
    but uses characters per word instead of syllables like :func:`coleman_liau_index()`.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/Automated_readability_index
    """
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_chars = basics.n_chars(words)
    return (4.71 * n_chars / n_words) + (0.5 * n_words / n_sents) - 21.43


def automatic_arabic_readability_index(doc: Doc) -> float:
    """
    Readability test for Arabic-language texts based on number of characters and
    average word and sentence lengths.

    Higher value => more difficult text.

    Args:
        doc

    References:
        Al Tamimi, Abdel Karim, et al. "AARI: automatic arabic readability index."
        Int. Arab J. Inf. Technol. 11.4 (2014): 370-378.
    """
    if doc.lang_ != "ar":
        LOGGER.warning(
            "doc lang = '%s', but this readability statistic is intended for use on "
            "'ar'-language texts only",
            doc.lang_,
        )
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_chars = basics.n_chars(words)
    return (3.28 * n_chars) + (1.43 * n_chars / n_words) + (1.24 * n_words / n_sents)


def coleman_liau_index(doc: Doc) -> float:
    """
    Readability test whose value estimates the number of years of education
    required to understand a text, similar to :func:`flesch_kincaid_grade_level()`
    and :func:`smog_index()`, but using characters per word instead of syllables.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
    """
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_chars = basics.n_chars(words)
    return (5.879851 * n_chars / n_words) - (29.587280 * n_sents / n_words) - 15.800804


def flesch_kincaid_grade_level(doc: Doc) -> float:
    """
    Readability test used widely in education, whose value estimates the U.S.
    grade level / number of years of education required to understand a text.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch.E2.80.93Kincaid_grade_level
    """  # noqa: E501
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_syllables = basics.n_syllables(words, lang=doc.lang_)
    return (11.8 * n_syllables / n_words) + (0.39 * n_words / n_sents) - 15.59


def flesch_reading_ease(doc: Doc, *, lang: Optional[str] = None) -> float:
    """
    Readability test used as a general-purpose standard in several languages, based on
    a weighted combination of avg. sentence length and avg. word length. Values usually
    fall in the range [0, 100], but may be arbitrarily negative in extreme cases.

    Higher value => easier text.

    Args:
        doc
        lang

    Note:
        Coefficients in this formula are language-dependent;
        if ``lang`` is null, the value of ``Doc.lang_`` is used.

    References:
        English: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch_reading_ease
        German: https://de.wikipedia.org/wiki/Lesbarkeitsindex#Flesch-Reading-Ease
        Spanish: Fernández-Huerta formulation
        French: ?
        Italian: https://it.wikipedia.org/wiki/Formula_di_Flesch
        Dutch: ?
        Portuguese: https://pt.wikipedia.org/wiki/Legibilidade_de_Flesch
        Turkish: Atesman formulation
        Russian: https://ru.wikipedia.org/wiki/%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81_%D1%83%D0%B4%D0%BE%D0%B1%D0%BE%D1%87%D0%B8%D1%82%D0%B0%D0%B5%D0%BC%D0%BE%D1%81%D1%82%D0%B8
    """  # noqa: E501
    if lang is None:
        lang = doc.lang_
    try:
        coefs = _FRE_COEFS[lang]
    except KeyError:
        raise ValueError(errors.value_invalid_msg("lang", lang, list(_FRE_COEFS.keys())))

    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_syllables = basics.n_syllables(words, lang=lang)
    return (
        coefs["base"]
        - (coefs["asl"] * n_words / n_sents)
        - (coefs["awl"] * n_syllables / n_words)
    )


def gulpease_index(doc: Doc) -> float:
    """
    Readability test for Italian-language texts, whose value is in the range [0, 100]
    similar to :func:`flesch_reading_ease()`.

    Higher value => easier text.

    Args:
        doc

    References:
        https://it.wikipedia.org/wiki/Indice_Gulpease
    """
    if doc.lang_ != "it":
        LOGGER.warning(
            "doc lang = '%s', but this readability statistic is intended for use on "
            "'it'-language texts only",
            doc.lang_,
        )
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_chars = basics.n_chars(words)
    return (300 * n_sents / n_words) - (10 * n_chars / n_words) + 89


def gunning_fog_index(doc: Doc) -> float:
    """
    Readability test whose value estimates the number of years of education
    required to understand a text, similar to :func:`flesch_kincaid_grade_level()`
    and :func:`smog_index()`.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/Gunning_fog_index
    """
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_polysyllable_words = basics.n_polysyllable_words(words, lang=doc.lang_)
    return 0.4 * ((n_words / n_sents) + (100 * n_polysyllable_words / n_words))


def lix(doc: Doc) -> float:
    """
    Readability test commonly used in Sweden on both English- and non-English-language
    texts, whose value estimates the difficulty of reading a foreign text.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/Lix_(readability_test)
    """
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_long_words = basics.n_long_words(words)
    return (n_words / n_sents) + (100 * n_long_words / n_words)


def mu_legibility_index(doc: Doc) -> float:
    """
    Readability test for Spanish-language texts based on number of words and
    the mean and variance of their lengths in characters, whose value is in the range
    [0, 100].

    Higher value => easier text.

    Args:
        doc

    References:
        Muñoz, M., and J. Muñoz. "Legibilidad Mµ." Viña del Mar: CHL (2006).
    """
    if doc.lang_ != "es":
        LOGGER.warning(
            "doc lang = '%s', but this readability statistic is intended for use on "
            "'es'-language texts only",
            doc.lang_,
        )
    n_chars_per_word = basics.n_chars_per_word(doc)
    n_words = len(n_chars_per_word)
    if n_words < 2:
        LOGGER.warning(
            "mu legibility index is undefined for texts with fewer than two words; "
            "returning 0.0"
        )
        return 0.0
    return (
        100
        * (n_words / (n_words - 1))
        * (statistics.mean(n_chars_per_word) / statistics.variance(n_chars_per_word))
    )


def perspicuity_index(doc: Doc) -> float:
    """
    Readability test for Spanish-language texts, whose value is in the range [0, 100];
    very similar to the Spanish-specific formulation of :func:`flesch_reading_ease()`,
    but included additionally since it's become a common readability standard.

    Higher value => easier text.

    Args:
        doc

    References:
        Pazos, Francisco Szigriszt. Sistemas predictivos de legibilidad del mensaje
        escrito: fórmula de perspicuidad. Universidad Complutense de Madrid,
        Servicio de Reprografía, 1993.
    """
    if doc.lang_ != "es":
        LOGGER.warning(
            "doc lang = '%s', but this readability statistic is intended for use on "
            "'es'-language texts only",
            doc.lang_,
        )
    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_syllables = basics.n_syllables(words, lang=doc.lang_)
    return 206.835 - (n_words / n_sents) - (62.3 * (n_syllables / n_words))


def smog_index(doc: Doc) -> float:
    """
    Readability test commonly used in medical writing and the healthcare industry,
    whose value estimates the number of years of education required to understand a text
    similar to :func:`flesch_kincaid_grade_level()` and intended as a substitute for
    :func:`gunning_fog_index()`.

    Higher value => more difficult text.

    Args:
        doc

    References:
        https://en.wikipedia.org/wiki/SMOG
    """
    n_sents = basics.n_sents(doc)
    n_polysyllable_words = basics.n_polysyllable_words(doc, lang=doc.lang_)
    if n_sents < 30:
        LOGGER.warning("SMOG index may be unreliable for n_sents < 30")
    return (1.0430 * math.sqrt(30 * n_polysyllable_words / n_sents)) + 3.1291


def wiener_sachtextformel(doc: Doc, *, variant: int = 1) -> float:
    """
    Readability test for German-language texts, whose value estimates the grade
    level required to understand a text.

    Higher value => more difficult text.

    Args:
        doc
        variant

    References:
        https://de.wikipedia.org/wiki/Lesbarkeitsindex#Wiener_Sachtextformel
    """
    if doc.lang_ != "de":
        LOGGER.warning(
            "doc lang = '%s', but this readability statistic is intended for use on "
            "'de'-language texts only",
            doc.lang_,
        )

    words = tuple(utils.get_words(doc))
    n_sents = basics.n_sents(doc)
    n_words = basics.n_words(words)
    n_long_words = basics.n_long_words(words)
    n_polysyllable_words = basics.n_polysyllable_words(words, lang=doc.lang_)
    n_monosyllable_words = basics.n_monosyllable_words(words, lang=doc.lang_)

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
        raise ValueError(errors.value_invalid_msg("variant", variant, [1, 2, 3, 4]))
