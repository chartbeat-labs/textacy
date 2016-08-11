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

from ftfy import fix_text
from unidecode import unidecode

from textacy.constants import (CURRENCIES, URL_REGEX, SHORT_URL_REGEX, EMAIL_REGEX,
                               PHONE_REGEX, NUMBERS_REGEX, PUNCT_REGEX, CURRENCY_REGEX,
                               LINEBREAK_REGEX, NONBREAKING_SPACE_REGEX)


def fix_bad_unicode(text, normalization='NFC'):
    """
    Fix unicode text that's "broken" using `ftfy <http://ftfy.readthedocs.org/>`_;
    this includes mojibake, HTML entities and other code cruft,
    and non-standard forms for display purposes.

    Args:
        text (str): raw text
        normalization ({'NFC', 'NFKC', 'NFD', 'NFKD'}): if 'NFC',
            combines characters and diacritics written using separate code points,
            e.g. converting "e" plus an acute accent modifier into "é"; unicode
            can be converted to NFC form without any change in its meaning!
            if 'NFKC', additional normalizations are applied that can change
            the meanings of characters, e.g. ellipsis characters will be replaced
            with three periods

    Returns:
        str
    """
    return fix_text(text, normalization=normalization)


def transliterate_unicode(text):
    """
    Try to represent unicode data in ascii characters similar to what a human
    with a US keyboard would choose.

    Works great for languages of Western origin, worse the farther the language
    gets from Latin-based alphabets. It's based on hand-tuned character mappings
    that also contain ascii approximations for symbols and non-Latin alphabets.
    """
    return unidecode(text)


def normalize_whitespace(text):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more linebreaks with a single newline. Also strip leading/trailing whitespace.
    """
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()


def unpack_contractions(text):
    """
    Replace *English* contractions in ``text`` str with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.
    """
    # standard
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
    return text


def replace_urls(text, replace_with='*URL*'):
    """Replace all URLs in ``text`` str with ``replace_with`` str."""
    return URL_REGEX.sub(replace_with, SHORT_URL_REGEX.sub(replace_with, text))


def replace_emails(text, replace_with='*EMAIL*'):
    """Replace all emails in ``text`` str with ``replace_with`` str."""
    return EMAIL_REGEX.sub(replace_with, text)


def replace_phone_numbers(text, replace_with='*PHONE*'):
    """Replace all phone numbers in ``text`` str with ``replace_with`` str."""
    return PHONE_REGEX.sub(replace_with, text)


def replace_numbers(text, replace_with='*NUMBER*'):
    """Replace all numbers in ``text`` str with ``replace_with`` str."""
    return NUMBERS_REGEX.sub(replace_with, text)


def remove_punct(text):
    """Remove all punctuation from ``text`` str (replace punct marks with empty string)."""
    return PUNCT_REGEX.sub('', text)


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
        for k, v in CURRENCIES.items():
            text = text.replace(k, v)
        return text
    else:
        return CURRENCY_REGEX.sub(replace_with, text)


def remove_accents(text, method='unicode'):
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
    if method == 'unicode':
        return ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if not unicodedata.combining(c))
    elif method == 'ascii':
        return unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii')
    else:
        msg = '`method` must be either "unicode" and "ascii", not {}'.format(method)
        raise ValueError(msg)


def preprocess_text(text, fix_unicode=False, lowercase=False, transliterate=False,
                    no_urls=False, no_emails=False, no_phone_numbers=False,
                    no_numbers=False, no_currency_symbols=False, no_punct=False,
                    no_contractions=False, no_accents=False):
    """
    Normalize various aspects of a raw text doc before parsing it with Spacy.
    A convenience function for applying all other preprocessing functions in one go.

    Args:
        text (str): raw text to preprocess
        fix_unicode (bool): if True, fix "broken" unicode such as
            mojibake and garbled HTML entities
        lowercase (bool): if True, all text is lower-cased
        transliterate (bool): if True, convert non-ascii characters
            into their closest ascii equivalents
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
            with unaccented versions; NB: if `transliterate` is True, this option
            is redundant

    Returns:
        str: input ``text`` processed according to function args

    .. warning:: These changes may negatively affect subsequent NLP analysis
        performed on the text, so choose carefully, and preprocess at your own
        risk!
    """
    if fix_unicode is True:
        text = fix_bad_unicode(text, normalization='NFC')
    if transliterate is True:
        text = transliterate_unicode(text)
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
        text = remove_accents(text, method='unicode')
    if no_punct is True:
        text = remove_punct(text)
    if lowercase is True:
        text = text.lower()
    # always normalize whitespace; treat linebreaks separately from spacing
    text = normalize_whitespace(text)

    return text
