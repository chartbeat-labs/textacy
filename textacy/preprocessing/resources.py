# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import sys
import unicodedata

from .. import compat

# compile regexes, so we don't do this on the fly and rely on caching

RE_LINEBREAK = re.compile(r"((\r\n)|[\n\v])+")
RE_NONBREAKING_SPACE = re.compile(r"(?!\n)\s+", flags=re.UNICODE)

# source: https://gist.github.com/dperini/729294
RE_URL = re.compile(
    r"(?:^|(?<![\w/.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?://|ftp://|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?"
    r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
    r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:/\S*)?"
    r"(?:$|(?![\w?!+&/]))",
    flags=re.UNICODE | re.IGNORECASE)

RE_SHORT_URL = re.compile(
    r"(?:^|(?<![\w/.]))"
    # optional scheme
    r"(?:(?:https?://)?)"
    # domain
    r"(?:\w-?)*?\w+(?:\.[a-z]{2,12}){1,3}"
    r"/"
    # hash
    r"[^\s.,?!'\"|+]{2,12}"
    r"(?:$|(?![\w?!+&/]))",
    flags=re.UNICODE | re.IGNORECASE)

RE_EMAIL = re.compile(
    r"(?:mailto:)?"
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}"
    r"(?:$|(?=\b))",
    flags=re.UNICODE | re.IGNORECASE)

RE_USER_HANDLE = re.compile(
    r"(?:^|(?<![\w@.]))@\w+",
    flags=re.UNICODE | re.IGNORECASE)

RE_HASHTAG = re.compile(
    r"(?:^|(?<![\w#＃.]))(#|＃)(?!\d)\w+",
    flags=re.UNICODE | re.IGNORECASE)

RE_PHONE_NUMBER = re.compile(
    # core components of a phone number
    r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})"
    # extensions, etc.
    r"(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))",
    flags=re.UNICODE | re.IGNORECASE)

RE_NUMBER = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))")

RE_CURRENCY_SYMBOL = re.compile(
    r"[$¢£¤¥ƒ֏؋৲৳૱௹฿៛ℳ元円圆圓﷼\u20A0-\u20C0]",
    flags=re.UNICODE)

if compat.is_narrow_unicode:
    RE_EMOJI = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF]",
        flags=re.UNICODE | re.IGNORECASE)
else:
    RE_EMOJI = re.compile(
        r"[\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]",
        flags=re.UNICODE | re.IGNORECASE)

RE_HYPHENATED_WORD = re.compile(
    r"(\w{2,}(?<!\d))\-\s+((?!\d)\w{2,})",
    flags=re.UNICODE | re.IGNORECASE)


# build mapping of unicode punctuation symbol ordinals to their replacements
# and lazy-load the big one, since it's relatively expensive to compute

PUNCT_TRANSLATION_TABLE = None


def _get_punct_translation_table():
    global PUNCT_TRANSLATION_TABLE
    if PUNCT_TRANSLATION_TABLE is None:
        PUNCT_TRANSLATION_TABLE = dict.fromkeys(
            (i for i in compat.range_(sys.maxunicode)
             if unicodedata.category(compat.chr_(i)).startswith("P")),
            " "
        )
    return PUNCT_TRANSLATION_TABLE


QUOTE_TRANSLATION_TABLE = {
    ord(x): ord(y)
    for x, y in zip("‘’´`“”", "''''\"\"")}
