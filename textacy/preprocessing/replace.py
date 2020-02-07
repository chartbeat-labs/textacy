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


def replace_currency_symbols(text: str, replace_with: str = "_CUR_") -> str:
    """Replace all currency symbols in ``text`` with ``replace_with``."""
    return RE_CURRENCY_SYMBOL.sub(replace_with, text)


def replace_emails(text: str, replace_with: str = "_EMAIL_") -> str:
    """Replace all email addresses in ``text`` with ``replace_with``."""
    return RE_EMAIL.sub(replace_with, text)


def replace_emojis(text: str, replace_with: str = "_EMOJI_") -> str:
    """
    Replace all emoji and pictographs in ``text`` with ``replace_with``.

    Note:
        If your Python has a narrow unicode build ("USC-2"), only dingbats
        and miscellaneous symbols are replaced because Python isn't able
        to represent the unicode data for things like emoticons. Sorry!
    """
    return RE_EMOJI.sub(replace_with, text)


def replace_hashtags(text: str, replace_with: str = "_TAG_") -> str:
    """Replace all hashtags in ``text`` with ``replace_with``."""
    return RE_HASHTAG.sub(replace_with, text)


def replace_numbers(text: str, replace_with: str = "_NUMBER_") -> str:
    """Replace all numbers in ``text`` with ``replace_with``."""
    return RE_NUMBER.sub(replace_with, text)


def replace_phone_numbers(text: str, replace_with: str = "_PHONE_") -> str:
    """Replace all phone numbers in ``text`` with ``replace_with``."""
    return RE_PHONE_NUMBER.sub(replace_with, text)


def replace_urls(text: str, replace_with: str = "_URL_") -> str:
    """Replace all URLs in ``text`` with ``replace_with``."""
    return RE_SHORT_URL.sub(replace_with, RE_URL.sub(replace_with, text))


def replace_user_handles(text: str, replace_with: str = "_USER_") -> str:
    """Replace all user handles in ``text`` with ``replace_with``."""
    return RE_USER_HANDLE.sub(replace_with, text)
