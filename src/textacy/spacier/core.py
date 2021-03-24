"""
:mod:`textacy.spacier.core`: Convenient entry point for loading spaCy language pipelines
and making spaCy docs.
"""
import functools
import logging
import pathlib
from typing import Callable, Optional, Tuple, Union

import spacy
from cachetools import cached
from cachetools.keys import hashkey
from spacy.language import Language
from spacy.tokens import Doc

from .. import cache, errors, lang_id, utils


LOGGER = logging.getLogger(__name__)


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "spacy_lang"))
def load_spacy_lang(
    name: Union[str, pathlib.Path],
    disable: Optional[Tuple[str, ...]] = None,
    allow_blank: bool = False,
) -> Language:
    """
    Load a spaCy ``Language``: a shared vocabulary and language-specific data
    for tokenizing text, and (if available) model data and a processing pipeline
    containing a sequence of components for annotating a document.
    An LRU cache saves languages in memory for quick reloading.

    .. code-block:: pycon

        >>> en_nlp = textacy.load_spacy_lang("en")
        >>> en_nlp = textacy.load_spacy_lang("en_core_web_sm")
        >>> en_nlp = textacy.load_spacy_lang("en", disable=("parser",))
        >>> textacy.load_spacy_lang("ar")
        ...
        OSError: [E050] Can't find model 'ar'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
        >>> textacy.load_spacy_lang("ar", allow_blank=True)
        <spacy.lang.ar.Arabic at 0x126418550>

    Args:
        name: spaCy language to load.
            Could be a shortcut link, full package name, or path to model directory,
            or a 2-letter ISO language code for which spaCy has language data.
        disable: Names of pipeline components to disable, if any.

            .. note:: Although spaCy's API specifies this argument as a list,
               here we require a tuple. Pipelines are stored in the LRU cache
               with unique identifiers generated from the hash of the function
               name and args --- and lists aren't hashable.

        allow_blank: If True, allow loading of blank spaCy ``Language`` s;
            if False, raise an OSError if a full processing pipeline isn't available.
            Note that spaCy ``Doc`` s produced by blank languages are missing
            key functionality, e.g. POS tags, entities, sentences.

    Returns:
        A loaded spaCy ``Language``.

    Raises:
        OSError
        ImportError

    See Also:
        * https://spacy.io/api/top-level#spacy.load
        * https://spacy.io/api/top-level#spacy.blank
    """
    if disable is None:
        disable = []
    # load a full spacy lang processing pipeline
    try:
        spacy_lang = spacy.load(_get_full_model_name(name), disable=disable)
        LOGGER.info("loaded '%s' spaCy language pipeline", name)
        return spacy_lang
    except OSError as e:
        # fall back to a blank spacy lang
        if allow_blank is True and isinstance(name, str) and len(name) == 2:
            spacy_lang = spacy.blank(name)
            LOGGER.warning("loaded '%s' spaCy language blank", name)
            return spacy_lang
        else:
            raise e


def _get_full_model_name(name: Union[str, pathlib.Path]) -> Union[str, pathlib.Path]:
    """This is a hack for spaCy v3. Hopefully temporary."""
    if isinstance(name, str) and len(name) == 2:
        candidate_model_names = [
            mn for mn in spacy.util.get_installed_models() if mn.startswith(name)
        ]
        if len(candidate_model_names) == 1:
            return candidate_model_names[0]
        else:
            LOGGER.error(
                f"unable to infer which spaCy model to load for lang='{name}'"
            )
    return name


def make_spacy_doc(
    data: Union[str, Tuple[str, dict], Doc],
    lang: Union[str, Callable[[str], str], Language] = lang_id.identify_lang,
) -> Doc:
    """
    Make a :class:`spacy.tokens.Doc` from valid inputs, and automatically
    load/validate :class:`spacy.language.Language` pipelines to process ``data``.

    Make a ``Doc`` from text:

    .. code-block:: pycon

        >>> text = "To be, or not to be, that is the question."
        >>> doc = make_spacy_doc(text)
        >>> doc._.preview
        'Doc(13 tokens: "To be, or not to be, that is the question.")'

    Make a ``Doc`` from a (text, metadata) pair, aka a "record":

    .. code-block:: pycon

        >>> record = (text, {"author": "Shakespeare, William"})
        >>> doc = make_spacy_doc(record)
        >>> doc._.preview
        'Doc(13 tokens: "To be, or not to be, that is the question.")'
        >>> doc._.meta
        {'author': 'Shakespeare, William'}

    Specify the language / ``Language`` pipeline used to process the text --- or don't:

    .. code-block:: pycon

        >>> make_spacy_doc(text)
        >>> make_spacy_doc(text, lang="en")
        >>> make_spacy_doc(text, lang="en_core_web_sm")
        >>> make_spacy_doc(text, lang=textacy.load_spacy_lang("en"))
        >>> make_spacy_doc(text, lang=textacy.identify_lang)

    Ensure that an already-processed ``Doc`` is compatible with ``lang``:

    .. code-block:: pycon

        >>> spacy_lang = textacy.load_spacy_lang("en")
        >>> doc = spacy_lang(text)
        >>> make_spacy_doc(doc, lang="en")
        >>> make_spacy_doc(doc, lang="es")
        ...
        ValueError: lang of spacy pipeline used to process document ('en') must be the same as `lang` ('es')

    Args:
        data: Make a :class:`spacy.tokens.Doc` from a text or (text, metadata) pair.
            If already a ``Doc``, ensure that it's compatible with ``lang``
            to avoid surprises downstream, and return it as-is.
        lang: Language with which spaCy processes (or processed) ``data``.

            *If known*, pass a standard 2-letter language code (e.g. "en"),
            or the name of a spacy language pipeline (e.g. "en_core_web_md"),
            or an already-instantiated :class:`spacy.language.Language` object.
            *If not known*, pass a function that takes unicode text as input
            and outputs a standard 2-letter language code.

            A given / detected language string is then used to instantiate
            a corresponding ``Language`` with all default components enabled.

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
        raise TypeError(
            errors.type_invalid_msg(
                "data", type(data), Union[str, Tuple[str, dict], Doc]
            )
        )


def _make_spacy_doc_from_text(
    text: str, lang: Union[str, Callable[[str], str], Language],
) -> Doc:
    if isinstance(lang, str):
        spacy_lang = load_spacy_lang(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(text)
        spacy_lang = load_spacy_lang(langstr)
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "lang", type(lang), Union[str, Callable[[str], str], Language]
            )
        )
    return spacy_lang(text)


def _make_spacy_doc_from_record(
    record: Tuple[str, dict], lang: Union[str, Callable[[str], str], Language],
) -> Doc:
    if isinstance(lang, str):
        spacy_lang = load_spacy_lang(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(record[0])
        spacy_lang = load_spacy_lang(langstr)
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "lang", type(lang), Union[str, Callable[[str], str], Language]
            )
        )
    doc = spacy_lang(record[0])
    doc._.meta = record[1]
    return doc


def _make_spacy_doc_from_doc(
    doc: Doc, lang: Union[str, Callable[[str], str], Language],
) -> Doc:
    # these checks are probably unnecessary, but in case a user
    # has done something strange, we should complain...
    if isinstance(lang, str):
        # a `lang` as str could be a specific spacy model name,
        # e.g. "en_core_web_sm", while `langstr` would only be "en"
        langstr = doc.vocab.lang
        if not lang.startswith(langstr):
            raise ValueError(
                f"lang of spacy pipeline used to process document ('{langstr}') "
                f"must be the same as `lang` ('{lang}')"
            )
    elif isinstance(lang, spacy.language.Language):
        # just want to make sure that doc and lang share the same vocabulary
        if doc.vocab is not lang.vocab:
            raise ValueError(
                f"`spacy.vocab.Vocab` used to process document ({doc.vocab}) "
                f"must be the same as that used by the `lang` pipeline ({lang.vocab})"
            )
    elif callable(lang) is False:
        # there's nothing to be done with a callable lang, since we already have
        # the doc, and checking the text lang is an unnecessary performance hit
        raise TypeError(
            errors.type_invalid_msg(
                "lang", type(lang), Union[str, Callable[[str], str], Language]
            )
        )
    return doc
