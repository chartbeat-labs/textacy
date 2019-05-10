# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import types

import spacy

from . import cache
from . import compat
from . import text_utils
from . import utils


def make_spacy_doc(data, lang=text_utils.detect_language):
    """
    TODO

    Args:
        data (str or Tuple[str, dict] or :class:`spacy.tokens.Doc`)
        lang (str or :class:`spacy.language.Language` or Callable)

    Returns:
        :class:`spacy.tokens.Doc`
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
        spacy_lang = cache.load_spacy(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(text)
        spacy_lang = cache.load_spacy(langstr)
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
        spacy_lang = cache.load_spacy(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    elif callable(lang):
        langstr = lang(record[0])
        spacy_lang = cache.load_spacy(langstr)
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
                "must be the same as `lang` ('{}')".format(lang, langstr)
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
