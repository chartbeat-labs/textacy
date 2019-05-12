# -*- coding: utf-8 -*-
"""
Functions that modify raw text *in-place*, replacing contractions, URLs, emails,
phone numbers, and currency symbols with standardized forms. These should be
applied before processing by `Spacy <http://spacy.io>`_, but be warned: preprocessing
may affect the interpretation of the text -- and spacy's processing of it.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import unicodedata

from . import constants


def fix_bad_unicode(text, normalization="NFC"):
    """
    Fix unicode text that's "broken" using `ftfy <https://ftfy.readthedocs.io>`_;
    this includes mojibake, HTML entities and other code cruft,
    and non-standard forms for display purposes.

    Warning:
        As of v0.7.0, this is no longer implemented within textacy. Instead,
        install and import ``ftfy`` independently, then call ``ftfy.fix_text(text)``
        with a much larger variety of params, as needed for your use case.
    """
    return NotImplementedError(
        "As of v0.7.0, :func:`fix_bad_unicode()` is no longer implemented in textacy. "
        "Instead, install and import ``ftfy`` directly, and call ``ftfy.fix_text(text)`` ,"
        "which is more extensive and customizable than textacy's wrapper of it."
        "For details, check out https://ftfy.readthedocs.io."
    )


def normalize_whitespace(text):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more linebreaks with a single newline. Also strip leading/trailing whitespace.
    """
    return constants.RE_NONBREAKING_SPACE.sub(
        " ", constants.RE_LINEBREAK.sub(r"\n", text)
    ).strip()


def unpack_contractions(text):
    """
    Replace *English* contractions in ``text`` str with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.
    """
    # standard
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't",
        r"\1\2 not",
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll",
        r"\1\2 will",
        text,
    )
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",
        r"\1\2 have",
        text,
    )
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text


def replace_urls(text, replace_with="*URL*"):
    """Replace all URLs in ``text`` str with ``replace_with`` str."""
    return constants.RE_URL.sub(
        replace_with, constants.RE_SHORT_URL.sub(replace_with, text)
    )


def replace_emails(text, replace_with="*EMAIL*"):
    """Replace all emails in ``text`` str with ``replace_with`` str."""
    return constants.RE_EMAIL.sub(replace_with, text)


def replace_phone_numbers(text, replace_with="*PHONE*"):
    """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
    return constants.RE_PHONE.sub(replace_with, text)


def replace_numbers(text, replace_with="*NUMBER*"):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return constants.RE_NUMBERS.sub(replace_with, text)


def replace_currency_symbols(text, replace_with=None):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.

    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")

    Returns:
        str
    """
    if replace_with is None:
        for k, v in constants.CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return constants.RE_CURRENCY.sub(replace_with, text)


def remove_punct(text, marks=None):
    """
    Remove punctuation from ``text`` by replacing all instances of ``marks``
    with whitespace.

    Args:
        text (str): raw text
        marks (str): If specified, remove only the characters in this string,
            e.g. ``marks=',;:'`` removes commas, semi-colons, and colons.
            Otherwise, all punctuation marks are removed.

    Returns:
        str

    Note:
        When ``marks=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise, a regular expression is used
        instead. The former's performance is about 5-10x faster.
    """
    if marks:
        return re.sub("[{}]+".format(re.escape(marks)), " ", text, flags=re.UNICODE)
    else:
        return text.translate(constants.PUNCT_TRANSLATE_UNICODE.data)


def remove_accents(text, method="unicode"):
    """
    Remove accents from any accented unicode characters in ``text`` str, either by
    transforming them into ascii equivalents or removing them entirely.

    Args:
        text (str): raw text
        method ({'unicode', 'ascii'}): if 'unicode', remove accented
            char for any unicode symbol with a direct ASCII equivalent; if 'ascii',
            remove accented char for any unicode symbol

            NB: the 'ascii' method is notably faster than 'unicode', but less good

    Returns:
        str

    Raises:
        ValueError: if ``method`` is not in {'unicode', 'ascii'}
    """
    if method == "unicode":
        return "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    elif method == "ascii":
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )
    else:
        msg = '`method` must be either "unicode" and "ascii", not {}'.format(method)
        raise ValueError(msg)


def preprocess_text(
    text,
    fix_unicode=False,
    lowercase=False,
    no_urls=False,
    no_emails=False,
    no_phone_numbers=False,
    no_numbers=False,
    no_currency_symbols=False,
    no_punct=False,
    no_contractions=False,
    no_accents=False,
):
    """
    Normalize various aspects of a raw text doc before parsing it with Spacy.
    A convenience function for applying all other preprocessing functions in one go.

    Args:
        text (str): raw text to preprocess
        fix_unicode (bool): if True, fix "broken" unicode such as
            mojibake and garbled HTML entities
        lowercase (bool): if True, all text is lower-cased
        no_urls (bool): if True, replace all URL strings with '*URL*'
        no_emails (bool): if True, replace all email strings with '*EMAIL*'
        no_phone_numbers (bool): if True, replace all phone number strings
            with '*PHONE*'
        no_numbers (bool): if True, replace all number-like strings
            with '*NUMBER*'
        no_currency_symbols (bool): if True, replace all currency symbols
            with their standard 3-letter abbreviations
        no_punct (bool): if True, remove all punctuation (replace with
            empty string)
        no_contractions (bool): if True, replace *English* contractions
            with their unshortened forms
        no_accents (bool): if True, replace all accented characters
            with unaccented versions

    Returns:
        str: input ``text`` processed according to function args

    Warning:
        These changes may negatively affect subsequent NLP analysis performed
        on the text, so choose carefully, and preprocess at your own risk!
    """
    if fix_unicode is True:
        text = fix_bad_unicode(text, normalization="NFC")
    if no_urls is True:
        text = replace_urls(text)
    if no_emails is True:
        text = replace_emails(text)
    if no_phone_numbers is True:
        text = replace_phone_numbers(text)
    if no_numbers is True:
        text = replace_numbers(text)
    if no_currency_symbols is True:
        text = replace_currency_symbols(text)
    if no_contractions is True:
        text = unpack_contractions(text)
    if no_accents is True:
        text = remove_accents(text, method="unicode")
    if no_punct is True:
        text = remove_punct(text)
    if lowercase is True:
        text = text.lower()
    # always normalize whitespace; treat linebreaks separately from spacing
    text = normalize_whitespace(text)

    return text
