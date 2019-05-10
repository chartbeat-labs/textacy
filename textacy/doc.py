# -*- coding: utf-8 -*-
# TODO: module docstring
from __future__ import absolute_import, division, print_function, unicode_literals

import types

import spacy
from cytoolz import itertoolz

from . import cache
from . import compat
from . import text_utils
from . import utils
from .spacier.doc_extensions import *


_doc_extensions = {
    # property extensions
    "lang": {"getter": get_lang},
    "preview": {"getter": get_preview},
    "tokens": {"getter": get_tokens},
    "meta": {"getter": get_meta, "setter": set_meta},
    "n_tokens": {"getter": get_n_tokens},
    "n_sents": {"getter": get_n_sents},
    # method extensions
    "to_tokenized_text": {"method": to_tokenized_text},
    "to_tagged_text": {"method": to_tagged_text},
    "to_terms_list": {"method": to_terms_list},
    "to_bag_of_terms": {"method": to_bag_of_terms},
    "to_bag_of_words": {"method": to_bag_of_words},
    "to_semantic_network": {"method": to_semantic_network},
}


def set_doc_extensions():
    """
    Set textacy's custom property and method doc extensions
    on the global :class:`spacy.tokens.Doc`.
    """
    for name, kwargs in _doc_extensions.items():
        if not spacy.tokens.Doc.has_extension(name):
            spacy.tokens.Doc.set_extension(name, **kwargs)


def remove_doc_extensions():
    """
    Remove textacy's custom property and method doc extensions
    from the global :class:`spacy.tokens.Doc`.
    """
    for name in _doc_extensions.keys():
        _ = spacy.tokens.Doc.set_extension(name)


def make_spacy_doc(data, lang=text_utils.detect_language):
    """
    TODO

    Args:
        data (str or Tuple[str, dict])
        lang (str or :class:`spacy.language.Language` or Callable)

    Returns:
        :class:`spacy.tokens.Doc`
    """
    if isinstance(data, compat.unicode_):
        return _make_spacy_doc_from_text(data, lang)
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
    if callable(lang):
        langstr = lang(text)
        spacy_lang = cache.load_spacy(langstr)
    elif isinstance(lang, compat.unicode_):
        spacy_lang = cache.load_spacy(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
    else:
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language, types.FunctionType},
                type(lang),
            )
        )
    return spacy_lang(text)


def _make_spacy_doc_from_record(record, lang):
    if callable(lang):
        langstr = lang(text)
        spacy_lang = cache.load_spacy(langstr)
    elif isinstance(lang, compat.unicode_):
        spacy_lang = cache.load_spacy(lang)
        langstr = spacy_lang.lang
    elif isinstance(lang, spacy.language.Language):
        spacy_lang = lang
        langstr = spacy_lang.lang
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
