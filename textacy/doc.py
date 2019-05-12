# -*- coding: utf-8 -*-
"""A convenient and flexible entry point for making spaCy docs, one at a time."""
from __future__ import absolute_import, division, print_function, unicode_literals

import types

import spacy

from . import cache
from . import compat
from . import lang_utils
from . import utils


def make_spacy_doc(data, lang=lang_utils.detect_lang):
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
        >>> make_spacy_doc(text, lang=textacy.lang_utils.detect_lang)

    Ensure that an already-processed ``Doc`` is compatible with ``lang``:

    .. code-block:: pycon

        >>> spacy_lang = textacy.load_spacy_lang("en")
        >>> doc = spacy_lang(text)
        >>> make_spacy_doc(doc, lang="en")
        >>> make_spacy_doc(doc, lang="es")
        ...
        ValueError: lang of spacy pipeline used to process document ('en') must be the same as `lang` ('es')

    Args:
        data (str or Tuple[str, dict] or :class:`spacy.tokens.Doc`):
            Make a :class:`spacy.tokens.Doc` from a text or (text, metadata) pair.
            If already a ``Doc``, ensure that it's compatible with ``lang``
            to avoid surprises downstream, and return it as-is.
        lang (str or :class:`spacy.language.Language` or Callable):
            Language with which spaCy processes (or processed) ``data``.

            *If known*, pass a standard 2-letter language code (e.g. "en"),
            or the name of a spacy language pipeline (e.g. "en_core_web_md"),
            or an already-instantiated :class:`spacy.language.Language` object.
            *If not known*, pass a function that takes unicode text as input
            and outputs a standard 2-letter language code.

            A given / detected language string is then used to instantiate
            a corresponding ``Language`` with all default components enabled.

    Returns:
        :class:`spacy.tokens.Doc`

    Raises:
        TypeError
        ValueError
    """
    if isinstance(data, compat.unicode_):
        return _make_spacy_doc_from_text(data, lang)
    elif isinstance(data, spacy.tokens.Doc):
        return _make_spacy_doc_from_doc(data, lang)
    elif utils.is_record(data):
        return _make_spacy_doc_from_record(data, lang)
    else:
        raise TypeError(
            "`data` must be {}, not {}".format(
                {compat.unicode_, tuple},
                type(data),
            )
        )


def _make_spacy_doc_from_text(text, lang):
    if isinstance(lang, compat.unicode_):
        spacy_lang = cache.load_spacy_lang(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(text)
        spacy_lang = cache.load_spacy_lang(langstr)
    else:
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language, types.FunctionType},
                type(lang),
            )
        )
    return spacy_lang(text)


def _make_spacy_doc_from_record(record, lang):
    if isinstance(lang, compat.unicode_):
        spacy_lang = cache.load_spacy_lang(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(record[0])
        spacy_lang = cache.load_spacy_lang(langstr)
    else:
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language, types.FunctionType},
                type(lang),
            )
        )
    doc = spacy_lang(record[0])
    doc._.meta = record[1]
    return doc


def _make_spacy_doc_from_doc(doc, lang):
    # these checks are probably unnecessary, but in case a user
    # has done something strange, we should complain...
    if isinstance(lang, compat.unicode_):
        # a `lang` as str could be a specific spacy model name,
        # e.g. "en_core_web_sm", while `langstr` would only be "en"
        langstr = doc.vocab.lang
        if not lang.startswith(langstr):
            raise ValueError(
                "lang of spacy pipeline used to process document ('{}') "
                "must be the same as `lang` ('{}')".format(langstr, lang)
            )
    elif isinstance(lang, spacy.language.Language):
        # just want to make sure that doc and lang share the same vocabulary
        if doc.vocab is not lang.vocab:
            raise ValueError(
                "`spacy.vocab.Vocab` used to process document ('{}') "
                "must be the same as that used by the `lang` pipeline ('{}')".format(
                    doc.vocab, lang.vocab)
            )
    elif callable(lang) is False:
        # there's nothing to be done with a callable lang, since we already have
        # the doc, and checking the text lang is an unnecessary performance hit
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language, types.FunctionType},
                type(lang),
            )
        )
    return doc
