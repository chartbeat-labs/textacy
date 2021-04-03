"""
:mod:`textacy.spacier.core`: Convenient entry point for loading spaCy language pipelines
and making spaCy docs.
"""
from __future__ import annotations

import functools
import logging
import pathlib

import spacy
from cachetools import cached
from cachetools.keys import hashkey
from spacy.language import Language
from spacy.tokens import Doc

from .. import cache, errors, types, utils


LOGGER = logging.getLogger(__name__)


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "spacy_lang"))
def load_spacy_lang(name: str | pathlib.Path, **kwargs) -> Language:
    """
    Load a spaCy ``Language`` — a shared vocabulary and language-specific data
    for tokenizing text, and (if available) model data and a processing pipeline
    containing a sequence of components for annotating a document — and cache results,
    for quick reloading as needed.

    Note that as of spaCy v3, for which pipeline aliases are no longer allowed,
    this function is just a convenient access point to underlying :func:`spacy.load()`.

    .. code-block:: pycon

        >>> en_nlp = textacy.load_spacy_lang("en_core_web_sm")
        >>> en_nlp = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
        >>> textacy.load_spacy_lang("ar")
        ...
        OSError: [E050] Can't find model 'ar'. It doesn't seem to be a Python package or a valid path to a data directory.

    Args:
        name: Name or path to the spaCy language pipeline to load.
        **kwargs

    Note:
        Although spaCy's API specifies some kwargs as ``List[str]``, here we require
        ``Tuple[str, ...]`` equivalents. Language pipelines are stored in an LRU cache
        with unique identifiers generated from the hash of the function name and args —
        and lists aren't hashable.

    Returns:
        Loaded spaCy ``Language``.

    Raises:
        OSError

    See Also:
        https://spacy.io/api/top-level#spacy.load
    """
    spacy_lang = spacy.load(name, **kwargs)
    LOGGER.info("loaded '%s' spaCy language pipeline", name)
    return spacy_lang


def make_spacy_doc(data: types.DocData, lang: types.LangLikeInContext) -> Doc:
    """
    Make a :class:`spacy.tokens.Doc` from valid inputs, and automatically
    load/validate :class:`spacy.language.Language` pipelines to process ``data``.

    Make a ``Doc`` from text:

    .. code-block:: pycon

        >>> text = "To be, or not to be, that is the question."
        >>> doc = make_spacy_doc(text, "en_core_web_sm")
        >>> doc._.preview
        'Doc(13 tokens: "To be, or not to be, that is the question.")'

    Make a ``Doc`` from a (text, metadata) pair, aka a "record":

    .. code-block:: pycon

        >>> record = (text, {"author": "Shakespeare, William"})
        >>> doc = make_spacy_doc(record, "en_core_web_sm")
        >>> doc._.preview
        'Doc(13 tokens: "To be, or not to be, that is the question.")'
        >>> doc._.meta
        {'author': 'Shakespeare, William'}

    Specify the language pipeline used to process the text in a few different ways:

    .. code-block:: pycon

        >>> make_spacy_doc(text, lang="en_core_web_sm")
        >>> make_spacy_doc(text, lang=textacy.load_spacy_lang("en_core_web_sm"))
        >>> make_spacy_doc(text, lang=lambda txt: "en_core_web_sm")

    Ensure that an already-processed ``Doc`` is compatible with ``lang``:

    .. code-block:: pycon

        >>> spacy_lang = textacy.load_spacy_lang("en_core_web_sm")
        >>> doc = spacy_lang(text)
        >>> make_spacy_doc(doc, lang="en_core_web_sm")
        >>> make_spacy_doc(doc, lang="es_core_news_sm")
        ...
        ValueError: lang of spacy pipeline used to process document ('en_core_web_sm') must be the same as `lang` ('es_core_news_sm')

    Args:
        data: Make a :class:`spacy.tokens.Doc` from a text or (text, metadata) pair.
            If already a ``Doc``, ensure that it's compatible with ``lang``
            to avoid surprises downstream, and return it as-is.
        lang: Language with which spaCy processes (or processed) ``data``,
            represented as a full spaCy language pipeline name, an instantiated
            pipeline, or a callable function that takes the text component of ``data``
            and outputs an appropriate pipeline name or instance.

    Returns:
        Processed spaCy Doc.

    Raises:
        TypeError
        ValueError
    """
    if isinstance(data, str):
        return _make_spacy_doc_from_text(data, lang)
    elif isinstance(data, Doc):
        return _make_spacy_doc_from_doc(data, lang)
    elif utils.is_record(data):
        return _make_spacy_doc_from_record(data, lang)
    else:
        raise TypeError(errors.type_invalid_msg("data", type(data), types.DocData))


def _resolve_spacy_lang(text: str, lang: types.LangLikeInContext) -> Language:
    if isinstance(lang, str):
        return load_spacy_lang(lang)
    elif isinstance(lang, Language):
        return lang
    elif callable(lang):
        return _resolve_spacy_lang(text, lang(text))
    else:
        raise TypeError(
            errors.type_invalid_msg("lang", type(lang), types.LangLikeInContext)
        )


def _make_spacy_doc_from_text(text: str, lang: types.LangLikeInContext) -> Doc:
    spacy_lang = _resolve_spacy_lang(text, lang)
    doc = spacy_lang(text)
    return doc


def _make_spacy_doc_from_record(
    record: types.Record, lang: types.LangLikeInContext
) -> Doc:
    text, meta = record
    spacy_lang = _resolve_spacy_lang(text, lang)
    doc = spacy_lang(text)
    doc._.meta = meta
    return doc


def _make_spacy_doc_from_doc(doc: Doc, lang: types.LangLikeInContext) -> Doc:
    spacy_lang = _resolve_spacy_lang(doc.text, lang)
    # we want to make sure that the language used to create `doc` is the same as
    # the one passed here; however, the best we can do (bc of spaCy's API) is ensure
    # that they share the same vocab
    if doc.vocab is not spacy_lang.vocab:
        raise ValueError(
            f"`spacy.vocab.Vocab` used to process document ({doc.vocab}) "
            f"must be the same as that used by the `lang` pipeline ({spacy_lang.vocab})"
        )
    return doc
