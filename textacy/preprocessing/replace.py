"""
Replace
-------

Replace parts of raw text that are semantically important as members of a group
but not so much in the individual instances.
"""
from .resources import (
    RE_CURRENCY_SYMBOL,
    RE_EMAIL,
    RE_EMOJI,
    RE_HASHTAG,
    RE_NUMBER,
    RE_PHONE_NUMBER,
    RE_SHORT_URL,
    RE_URL,
    RE_USER_HANDLE,
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


def replace_emojis(text, replace_with="_EMOJI_"):
    """
    Replace all emoji and pictographs in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str

    Note:
        If your Python has a narrow unicode build ("USC-2"), only dingbats
        and miscellaneous symbols are replaced because Python isn't able
        to represent the unicode data for things like emoticons. Sorry!
    """
    return RE_EMOJI.sub(replace_with, text)


def replace_hashtags(text, replace_with="_TAG_"):
    """
    Replace all hashtags in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_HASHTAG.sub(replace_with, text)


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


def replace_user_handles(text, replace_with="_USER_"):
    """
    Replace all user handles in ``text`` with ``replace_with``.

    Args:
        text (str)
        replace_with (str)

    Returns:
        str
    """
    return RE_USER_HANDLE.sub(replace_with, text)
