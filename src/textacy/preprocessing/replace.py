"""
Replace
-------

:mod:`textacy.preprocessing.replace`: Replace parts of raw text that are semantically
important as members of a group but not so much in the individual instances. Can also
be used to remove such parts by specifying ``repl=""`` in function calls.
"""
from . import resources


def currency_symbols(text: str, repl: str = "_CUR_") -> str:
    """Replace all currency symbols in ``text`` with ``repl``."""
    return resources.RE_CURRENCY_SYMBOL.sub(repl, text)


def emails(text: str, repl: str = "_EMAIL_") -> str:
    """Replace all email addresses in ``text`` with ``repl``."""
    return resources.RE_EMAIL.sub(repl, text)


def emojis(text: str, repl: str = "_EMOJI_") -> str:
    """
    Replace all emoji and pictographs in ``text`` with ``repl``.

    Note:
        If your Python has a narrow unicode build ("USC-2"), only dingbats
        and miscellaneous symbols are replaced because Python isn't able
        to represent the unicode data for things like emoticons. Sorry!
    """
    return resources.RE_EMOJI.sub(repl, text)


def hashtags(text: str, repl: str = "_TAG_") -> str:
    """Replace all hashtags in ``text`` with ``repl``."""
    return resources.RE_HASHTAG.sub(repl, text)


def numbers(text: str, repl: str = "_NUMBER_") -> str:
    """Replace all numbers in ``text`` with ``repl``."""
    return resources.RE_NUMBER.sub(repl, text)


def phone_numbers(text: str, repl: str = "_PHONE_") -> str:
    """Replace all phone numbers in ``text`` with ``repl``."""
    return resources.RE_PHONE_NUMBER.sub(repl, text)


def urls(text: str, repl: str = "_URL_") -> str:
    """Replace all URLs in ``text`` with ``repl``."""
    return resources.RE_SHORT_URL.sub(repl, resources.RE_URL.sub(repl, text))


def user_handles(text: str, repl: str = "_USER_") -> str:
    """Replace all (Twitter-style) user handles in ``text`` with ``repl``."""
    return resources.RE_USER_HANDLE.sub(repl, text)
