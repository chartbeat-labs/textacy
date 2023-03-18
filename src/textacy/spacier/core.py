"""
:mod:`textacy.spacier.core`: Convenient entry point for loading spaCy language pipelines
and making spaCy docs.
"""
from __future__ import annotations

import functools
import logging
import pathlib
from typing import Optional

import spacy
from cachetools import cached
from cachetools.keys import hashkey
from spacy.language import Language
from spacy.tokens import Doc

from .. import cache, errors, types, utils
from . import extensions
from . import utils as sputils


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


def make_spacy_doc(
    data: types.DocData,
    lang: types.LangLikeInContext,
    *,
    chunk_size: Optional[int] = None,
) -> Doc:
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
        ValueError: `spacy.Vocab` used to process document must be the same as that used by the `lang` pipeline ('es_core_news_sm')

    Args:
        data: Make a :class:`spacy.tokens.Doc` from a text or (text, metadata) pair.
            If already a ``Doc``, ensure that it's compatible with ``lang``
            to avoid surprises downstream, and return it as-is.
        lang: Language with which spaCy processes (or processed) ``data``,
            represented as the full name of a spaCy language pipeline, the path on disk
            to it, an already instantiated pipeline, or a callable function that takes
            the text component of ``data`` and outputs one of the above representations.
        chunk_size: Size of chunks in number of characters into which ``text`` will be
            split before processing each via spaCy and concatenating the results
            into a single ``Doc``.

            .. note:: This is intended as a workaround for processing very long texts,
               for which spaCy is unable to allocate enough RAM. For best performance,
               chunk size should be somewhere between 1e3 and 1e7 characters,
               depending on how much RAM you have available.

               Since chunking is done by *character*, chunks' boundaries likely
               won't respect natural language segmentation, and as a result
               spaCy's models may make mistakes on sentences/words that cross them.

    Returns:
        Processed spaCy Doc.

    Raises:
        TypeError
        ValueError
    """
    if isinstance(data, str):
        return _make_spacy_doc_from_text(data, lang, chunk_size)
    elif isinstance(data, Doc):
        return _make_spacy_doc_from_doc(data, lang)
    elif utils.is_record(data):
        return _make_spacy_doc_from_record(data, lang, chunk_size)
    else:
        raise TypeError(errors.type_invalid_msg("data", type(data), types.DocData))


def _make_spacy_doc_from_text(
    text: str, lang: types.LangLikeInContext, chunk_size: Optional[int]
) -> Doc:
    spacy_lang = sputils.resolve_langlikeincontext(text, lang)
    if chunk_size:
        doc = _make_spacy_doc_from_text_chunks(text, spacy_lang, chunk_size)
    else:
        doc = spacy_lang(text)
    return doc


def _make_spacy_doc_from_record(
    record: types.Record, lang: types.LangLikeInContext, chunk_size: Optional[int]
) -> Doc:
    text, meta = record
    spacy_lang = sputils.resolve_langlikeincontext(text, lang)
    if chunk_size:
        doc = _make_spacy_doc_from_text_chunks(text, spacy_lang, chunk_size)
    else:
        doc = spacy_lang(text)
    doc._.meta = meta
    return doc


def _make_spacy_doc_from_text_chunks(text: str, lang: Language, chunk_size: int) -> Doc:
    text_chunks = (text[i : i + chunk_size] for i in range(0, len(text), chunk_size))
    return Doc.from_docs(list(lang.pipe(text_chunks)))


def _make_spacy_doc_from_doc(doc: Doc, lang: types.LangLikeInContext) -> Doc:
    spacy_lang = sputils.resolve_langlikeincontext(doc.text, lang)
    # we want to make sure that the language used to create `doc` is the same as
    # the one passed here; however, the best we can do (bc of spaCy's API) is ensure
    # that they share the same vocab
    if doc.vocab is not spacy_lang.vocab:
        raise ValueError(
            f"`spacy.Vocab` used to process document ({doc.vocab}) must be the same "
            f"as that used by the `lang` pipeline ({spacy_lang.vocab})"
        )
    return doc


def get_doc_preview(doc: Doc) -> str:
    """
    Get a short preview of ``doc``, including the number of tokens and a snippet.
    Typically used as a custom extension, like ``doc._.preview`` .
    """
    snippet = doc.text[:50].replace("\n", " ")
    if len(snippet) == 50:
        snippet = snippet[:47] + "..."
    return f'Doc({len(doc)} tokens: "{snippet}")'


def get_doc_meta(doc: Doc) -> dict:
    """
    Get custom metadata added to ``doc`` .
    Typically used as a custom extension, like ``doc._.meta`` .
    """
    return doc.user_data.get("textacy", {}).get("meta", {})


def set_doc_meta(doc: Doc, value: dict) -> None:
    """
    Add custom metadata ``value`` to ``doc`` .
    Typically used as a custom extension, like ``doc._.meta = value`` .
    """
    if not isinstance(value, dict):
        raise TypeError(errors.type_invalid_msg("value", type(value), dict))
    try:
        doc.user_data["textacy"]["meta"] = value
    except KeyError:
        # TODO: confirm that this is the same. it is, right??
        doc.user_data["textacy"] = {"meta": value}


@extensions.doc_extensions_registry.register("spacier")
def _get_spacier_doc_extensions() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "preview": {"getter": get_doc_preview},  # type: ignore
        "meta": {"getter": get_doc_meta, "setter": set_doc_meta},  # type: ignore
    }
