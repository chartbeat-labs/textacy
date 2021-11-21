"""
:mod:`textacy.text_stats.api`: Compute basic and readability statistics of documents.
"""
import functools
import logging
from typing import Callable, Dict, Literal, Optional, Tuple

import pyphen
from cachetools import cached
from cachetools.keys import hashkey
from spacy.tokens import Doc, Token

from .. import cache, constants, errors, extract
from . import basics, diversity, morph, readability


LOGGER = logging.getLogger(__name__)

DiversityNameType = Literal["ttr", "log-ttr", "segmented-ttr", "mtld", "hdd"]
ReadabilityNameType = Literal[
    "automated-readability-index",
    "automatic-arabic-readability-index",
    "coleman-liau-index",
    "flesch-kincaid-grade-level",
    "flesch-reading-ease",
    "gulpease-index",
    "gunning-fog-index",
    "lix",
    "mu-legibility-index",
    "perspicuity-index",
    "smog-index",
    "wiener-sachtextformel",
]

_DIVERSITY_NAME_TO_FUNC: Dict[str, Callable[..., float]] = {
    "ttr": diversity.ttr,
    "log-ttr": diversity.log_ttr,
    "segmented-ttr": diversity.segmented_ttr,
    "mtld": diversity.mtld,
    "hdd": diversity.hdd,
}
_READABILITY_NAME_TO_FUNC_LANG: Dict[
    str, Tuple[Callable[["TextStats"], float], Optional[str]]
] = {
    "automated-readability-index": (
        lambda ts: readability.automated_readability_index(
            ts.n_chars, ts.n_words, ts.n_sents
        ),
        None,
    ),
    "automatic-arabic-readability-index": (
        lambda ts: readability.automatic_arabic_readability_index(
            ts.n_chars, ts.n_words, ts.n_sents
        ),
        "ar",
    ),
    "coleman-liau-index": (
        lambda ts: readability.coleman_liau_index(ts.n_chars, ts.n_words, ts.n_sents),
        None,
    ),
    "flesch-kincaid-grade-level": (
        lambda ts: readability.flesch_kincaid_grade_level(
            ts.n_syllables, ts.n_words, ts.n_sents
        ),
        None,
    ),
    "flesch-reading-ease": (
        lambda ts: readability.flesch_reading_ease(
            ts.n_syllables, ts.n_words, ts.n_sents, lang=ts.lang
        ),
        None,
    ),
    "gulpease-index": (
        lambda ts: readability.gulpease_index(ts.n_chars, ts.n_words, ts.n_sents),
        "it",
    ),
    "gunning-fog-index": (
        lambda ts: readability.gunning_fog_index(
            ts.n_words, ts.n_polysyllable_words, ts.n_sents
        ),
        None,
    ),
    "lix": (
        lambda ts: readability.lix(ts.n_words, ts.n_long_words, ts.n_sents),
        None,
    ),
    "mu-legibility-index": (
        lambda ts: readability.mu_legibility_index(ts.n_chars_per_word),
        "es",
    ),
    "perspicuity-index": (
        lambda ts: readability.perspicuity_index(ts.n_syllables, ts.n_words, ts.n_sents),
        "es",
    ),
    "smog-index": (
        lambda ts: readability.smog_index(ts.n_polysyllable_words, ts.n_sents),
        None,
    ),
    "wiener-sachtextformel": (
        lambda ts: readability.wiener_sachtextformel(
            ts.n_words,
            ts.n_polysyllable_words,
            ts.n_monosyllable_words,
            ts.n_long_words,
            ts.n_sents,
            variant=1,
        ),
        "de",
    ),
}


class TextStats:
    """
    Class to compute a variety of basic and readability statistics for a given doc,
    where each stat is a lazily-computed attribute.

    .. code-block:: pycon

        >>> text = next(textacy.datasets.CapitolWords().texts(limit=1))
        >>> doc = textacy.make_spacy_doc(text, lang="en_core_web_sm")
        >>> ts = textacy.text_stats.TextStats(doc)
        >>> ts.n_words
        136
        >>> ts.n_unique_words
        80
        >>> ts.entropy
        6.00420319027642
        >>> ts.flesch_kincaid_grade_level
        11.817647058823532
        >>> ts.flesch_reading_ease
        50.707745098039254

    Some stats vary by language or are designed for use with specific languages:

    .. code-block:: pycon

        >>> text = (
        ...     "Muchos años después, frente al pelotón de fusilamiento, "
        ...     "el coronel Aureliano Buendía había de recordar aquella tarde remota "
        ...     "en que su padre lo llevó a conocer el hielo."
        ... )
        >>> doc = textacy.make_spacy_doc(text, lang="es")
        >>> ts = textacy.text_stats.TextStats(doc)
        >>> ts.n_words
        28
        >>> ts.perspicuity_index
        56.46000000000002
        >>> ts.mu_legibility_index
        71.18644067796609

    Each of these stats have stand-alone functions in :mod:`textacy.text_stats.basics`
    and :mod:`textacy.text_stats.readability` with more detailed info and links
    in the docstrings -- when in doubt, read the docs!

    Args:
        doc: A text document tokenized and (optionally) sentence-segmented by spaCy.
    """

    def __init__(self, doc: Doc):
        self.doc = doc
        self.lang: str = doc.vocab.lang
        self.words: Tuple[Token, ...] = tuple(
            extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False)
        )
        self._n_sents: Optional[int] = None
        self._n_words: Optional[int] = None
        self._n_unique_words: Optional[int] = None
        self._n_long_words: Optional[int] = None
        self._n_chars_per_word: Optional[Tuple[int, ...]] = None
        self._n_chars: Optional[int] = None
        self._n_syllables_per_word: Optional[Tuple[int, ...]] = None
        self._n_syllables: Optional[int] = None
        self._n_monosyllable_words: Optional[int] = None
        self._n_polysyllable_words: Optional[int] = None
        self._entropy: Optional[float] = None

    @property
    def n_sents(self) -> int:
        """
        Number of sentences in document.

        See Also:
            :func:`textacy.text_stats.basics.n_sents()`
        """
        if self._n_sents is None:
            self._n_sents = basics.n_sents(self.doc)
        return self._n_sents

    @property
    def n_words(self) -> int:
        """
        Number of words in document.

        See Also:
            :func:`textacy.text_stats.basics.n_words()`
        """
        if self._n_words is None:
            self._n_words = basics.n_words(self.words)
        return self._n_words

    @property
    def n_unique_words(self) -> int:
        """
        Number of *unique* words in document.

        See Also:
            :func:`textacy.text_stats.basics.n_unique_words()`
        """
        if self._n_unique_words is None:
            self._n_unique_words = basics.n_unique_words(self.words)
        return self._n_unique_words

    @property
    def n_long_words(self) -> int:
        """
        Number of long words in document.

        See Also:
            :func:`textacy.text_stats.basics.n_long_words()`
        """
        # TODO: should we vary char threshold by lang?
        if self._n_long_words is None:
            self._n_long_words = basics.n_long_words(
                self.n_chars_per_word,
                min_n_chars=7,
            )
        return self._n_long_words

    @property
    def n_chars_per_word(self) -> Tuple[int, ...]:
        """
        Number of characters for each word in document.

        See Also:
            :func:`textacy.text_stats.basics.n_chars_per_word()`
        """
        if self._n_chars_per_word is None:
            self._n_chars_per_word = basics.n_chars_per_word(self.words)
        return self._n_chars_per_word

    @property
    def n_chars(self) -> int:
        """
        Total number of characters in document.

        See Also:
            :func:`textacy.text_stats.basics.n_chars()`
        """
        if self._n_chars is None:
            self._n_chars = basics.n_chars(self.n_chars_per_word)
        return self._n_chars

    @property
    def n_syllables_per_word(self) -> Tuple[int, ...]:
        """
        Number of syllables for each word in document.

        See Also:
            :func:`textacy.text_stats.basics.n_syllables_per_word()`
        """
        if self._n_syllables_per_word is None:
            self._n_syllables_per_word = basics.n_syllables_per_word(
                self.words,
                self.lang,
            )
        return self._n_syllables_per_word

    @property
    def n_syllables(self) -> int:
        """
        Total number of syllables in document.

        See Also:
            :func:`textacy.text_stats.basics.n_syllables()`
        """
        if self._n_syllables is None:
            self._n_syllables = basics.n_syllables(self.n_syllables_per_word)
        return self._n_syllables

    @property
    def n_monosyllable_words(self) -> int:
        """
        Number of monosyllobic words in document.

        See Also:
            :func:`textacy.text_stats.basics.n_monosyllable_words()`
        """
        if self._n_monosyllable_words is None:
            self._n_monosyllable_words = basics.n_monosyllable_words(
                self.n_syllables_per_word,
            )
        return self._n_monosyllable_words

    @property
    def n_polysyllable_words(self) -> int:
        """
        Number of polysyllobic words in document.

        See Also:
            :func:`textacy.text_stats.basics.n_polysyllable_words()`
        """
        # TODO: should we vary syllable threshold by lang?
        if self._n_polysyllable_words is None:
            self._n_polysyllable_words = basics.n_polysyllable_words(
                self.n_syllables_per_word,
                min_n_syllables=3,
            )
        return self._n_polysyllable_words

    @property
    def entropy(self) -> float:
        """
        Entropy of words in document.

        See Also:
            :func:`textacy.text_stats.basics.entropy()`
        """
        if self._entropy is None:
            self._entropy = basics.entropy(self.words)
        return self._entropy

    @property
    def automated_readability_index(self) -> float:
        """
        Readability test for English-language texts. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.automated_readability_index()`
        """
        return readability.automated_readability_index(
            self.n_chars,
            self.n_words,
            self.n_sents,
        )

    @property
    def automatic_arabic_readability_index(self) -> float:
        """
        Readability test for Arabic-language texts. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.automatic_arabic_readability_index()`
        """
        if self.lang != "ar":
            LOGGER.warning(
                "doc lang = '%s', but automatic arabic readability index is meant "
                "for use on Arabic-language texts, only"
            )
        return readability.automatic_arabic_readability_index(
            self.n_chars,
            self.n_words,
            self.n_sents,
        )

    @property
    def coleman_liau_index(self) -> float:
        """
        Readability test, not language-specific. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.coleman_liau_index()`
        """
        return readability.coleman_liau_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def flesch_kincaid_grade_level(self) -> float:
        """
        Readability test, not language-specific. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.flesch_kincaid_grade_level()`
        """
        return readability.flesch_kincaid_grade_level(
            self.n_syllables,
            self.n_words,
            self.n_sents,
        )

    @property
    def flesch_reading_ease(self) -> float:
        """
        Readability test with several language-specific formulations.
        Higher value => easier text.

        See Also:
            :func:`textacy.text_stats.readability.flesch_reading_ease()`
        """
        return readability.flesch_reading_ease(
            self.n_syllables, self.n_words, self.n_sents, lang=self.lang
        )

    @property
    def gulpease_index(self) -> float:
        """
        Readability test for Italian-language texts. Higher value => easier text.

        See Also:
            :func:`textacy.text_stats.readability.gulpease_index()`
        """
        if self.lang != "it":
            LOGGER.warning(
                "doc lang = '%s', but gulpease index is meant for use on "
                "Italian-language texts, only"
            )
        return readability.gulpease_index(self.n_chars, self.n_words, self.n_sents)

    @property
    def gunning_fog_index(self) -> float:
        """
        Readability test, not language-specific. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.gunning_fog_index()`
        """
        return readability.gunning_fog_index(
            self.n_words,
            self.n_polysyllable_words,
            self.n_sents,
        )

    @property
    def lix(self) -> float:
        """
        Readability test for both English- and non-English-language texts.
        Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.lix()`
        """
        return readability.lix(self.n_words, self.n_long_words, self.n_sents)

    @property
    def mu_legibility_index(self) -> float:
        """
        Readability test for Spanish-language texts. Higher value => easier text.

        See Also:
            :func:`textacy.text_stats.readability.mu_legibility_index()`
        """
        if self.lang != "es":
            LOGGER.warning(
                "doc lang = '%s', but mu legibility index is meant for use on "
                "Spanish-language texts, only"
            )
        return readability.mu_legibility_index(self.n_chars_per_word)

    @property
    def perspicuity_index(self) -> float:
        """
        Readability test for Spanish-language texts. Higher value => easier text.

        See Also:
            :func:`textacy.text_stats.readability.perspicuity_index()`
        """
        if self.lang != "es":
            LOGGER.warning(
                "doc lang = '%s', but perspicuity index is meant for use on "
                "Spanish-language texts, only"
            )
        return readability.perspicuity_index(
            self.n_syllables,
            self.n_words,
            self.n_sents,
        )

    @property
    def smog_index(self) -> float:
        """
        Readability test, not language-specific. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.smog_index()`
        """
        return readability.smog_index(self.n_polysyllable_words, self.n_sents)

    @property
    def wiener_sachtextformel(self) -> float:
        """
        Readability test for German-language texts. Higher value => more difficult text.

        See Also:
            :func:`textacy.text_stats.readability.wiener_sachtextformel()`
        """
        if self.lang != "es":
            LOGGER.warning(
                "doc lang = '%s', but wiener sachtextformel is meant for use on "
                "German-language texts, only"
            )
        return readability.wiener_sachtextformel(
            self.n_words,
            self.n_polysyllable_words,
            self.n_monosyllable_words,
            self.n_long_words,
            self.n_sents,
            variant=1,
        )

    @property
    def morph_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Number of times each value for a given morphological label appears in document.

        See Also:
            :func:`textacy.text_stats.morph.get_morph_label_counts()`
        """
        # NOTE: afaict there is absolutely no way to get the spacy language pipeline
        # used to produce a given document from the document itself
        # so, we can't get the lang-specific set of morph labels here
        # and instead just scan through all of the UD v2 default labels
        # then filter out those that don't have any values in the document
        # not ideal, but it's what we're stuck with
        mcs = {
            label: morph.get_morph_label_counts(label, self.doc)
            for label in constants.UD_V2_MORPH_LABELS
        }
        return {
            morph_label: value_counts
            for morph_label, value_counts in mcs.items()
            if value_counts
        }

    def readability(self, name: ReadabilityNameType) -> float:
        """
        Compute a measure of text readability using a method with specified ``name``.

        See Also:
            :mod:`textacy.text_stats.readability`
        """
        try:
            func, lang = _READABILITY_NAME_TO_FUNC_LANG[name]
            if lang and self.lang != lang:
                LOGGER.warning(
                    "doc lang = '%s', but '%s' readability is meant for use on "
                    "'%s'-language texts only",
                    self.lang,
                    name,
                    lang,
                )
        except KeyError:
            raise ValueError(
                errors.value_invalid_msg(
                    "name", name, sorted(_READABILITY_NAME_TO_FUNC_LANG.keys())
                )
            )
        return func(self)

    def diversity(self, name: DiversityNameType, **kwargs) -> float:
        """
        Compute a measure of lexical diversity using a method with specified ``name`` ,
        optionally specifying method variants and parameters.

        Higher values => higher lexical diversity.

        See Also:
            :mod:`textacy.text_stats.diversity`
        """
        try:
            func = _DIVERSITY_NAME_TO_FUNC[name]
        except KeyError:
            raise ValueError(
                errors.value_invalid_msg(
                    "name", name, sorted(_DIVERSITY_NAME_TO_FUNC.keys())
                )
            )
        return func(self.words, **kwargs)


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
