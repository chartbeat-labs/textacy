"""
:mod:`textacy.text_stats.api`: Compute a variety of text statistics for documents.
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
    Class to compute a variety of basic, readability, morphological, and lexical diversity
    statistics for a given document.

    .. code-block:: pycon

        >>> text = next(textacy.datasets.CapitolWords().texts(limit=1))
        >>> doc = textacy.make_spacy_doc(text, lang="en_core_web_sm")
        >>> ts = textacy.text_stats.TextStats(doc)
        >>> ts.n_words
        137
        >>> ts.n_unique_words
        81
        >>> ts.entropy
        6.02267943673824
        >>> ts.readability("flesch-kincaid-grade-level")
        11.40259124087591
        >>> ts.diversity("ttr")
        0.5912408759124088

    Some readability stats vary by language or are designed for use with
    specific languages:

    .. code-block:: pycon

        >>> text = (
        ...     "Muchos años después, frente al pelotón de fusilamiento, "
        ...     "el coronel Aureliano Buendía había de recordar aquella tarde remota "
        ...     "en que su padre lo llevó a conocer el hielo."
        ... )
        >>> doc = textacy.make_spacy_doc(text, lang="es_core_news_sm")
        >>> ts = textacy.text_stats.TextStats(doc)
        >>> ts.readability("perspicuity-index")
        56.46000000000002
        >>> ts.readability("mu-legibility-index")
        71.18644067796609

    Each of these stats have stand-alone functions in :mod:`textacy.text_stats.basics` ,
    :mod:`textacy.text_stats.readability` , and :mod:`textacy.text_stats.diversity`
    with more detailed info and links in the docstrings -- when in doubt, read the docs!

    Args:
        doc: A text document tokenized and (optionally) sentence-segmented by spaCy.
    """

    def __init__(self, doc: Doc):
        self.doc = doc
        self.lang: str = doc.lang_
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
            self._n_long_words = basics.n_long_words(self.words, min_n_chars=7)
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
            self._n_chars = basics.n_chars(self.words)
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
                self.words, lang=self.lang
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
            self._n_syllables = basics.n_syllables(self.words, lang=self.lang)
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
                self.words, lang=self.lang
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
                self.words, lang=self.lang, min_n_syllables=3
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

        Higher values => more difficult text for the following methods:

        - automated readability index
        - automatic arabic readability index
        - colman-liau index
        - flesch-kincaid grade level
        - gunning-fog index
        - lix
        - smog index
        - wiener-sachtextformel

        Higher values => less difficult text for the following methods:

        - flesch reading ease
        - gulpease index
        - mu legibility index
        - perspicuity index

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
