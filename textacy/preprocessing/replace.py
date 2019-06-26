from __future__ import absolute_import, division, print_function, unicode_literals

from .resources import (
    RE_CURRENCY_SYMBOL,
    RE_EMAIL,
    RE_NUMBER,
    RE_PHONE_NUMBER,
    RE_SHORT_URL,
    RE_URL,
)


def replace_currency_symbols(text, replace_with="_CUR_"):
    """
    Replace all currency symbols in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_CURRENCY_SYMBOL.sub(replace_with, text)


def replace_emails(text, replace_with="_EMAIL_"):
    """
    Replace all email addresses in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_EMAIL.sub(replace_with, text)


def replace_numbers(text, replace_with="_NUMBER_"):
    """
    Replace all numbers in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_NUMBER.sub(replace_with, text)


def replace_phone_numbers(text, replace_with="_PHONE_"):
    """
    Replace all phone numbers in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_PHONE_NUMBER.sub(replace_with, text)


def replace_urls(text, replace_with="_URL_"):
    """
    Replace all URLs in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_URL.sub(replace_with, RE_SHORT_URL.sub(replace_with, text))
