# -*- coding: utf-8 -*-
"""
Collection of regular expressions and other (small, generally useful) constants.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import sys
import unicodedata

from . import compat

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NUMERIC_ENT_TYPES = {"ORDINAL", "CARDINAL", "MONEY", "QUANTITY", "PERCENT", "TIME", "DATE"}
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS = {"aux", "auxpass", "neg"}

REPORTING_VERBS = {
    "according", "accuse", "acknowledge", "add", "admit", "agree",
    "allege", "announce", "argue", "ask", "assert", "believe", "blame",
    "charge", "cite", "claim", "complain", "concede", "conclude",
    "confirm", "contend", "criticize", "declare", "decline", "deny",
    "describe", "disagree", "disclose", "estimate", "explain", "fear",
    "hope", "insist", "maintain", "mention", "note", "observe", "order",
    "predict", "promise", "recall", "recommend", "reply", "report", "say",
    "state", "stress", "suggest", "tell", "testify", "think", "urge", "warn",
    "worry", "write"
}

CURRENCIES = {
    "$": "USD", "zł": "PLN", "£": "GBP", "¥": "JPY", "฿": "THB",
    "₡": "CRC", "₦": "NGN", "₩": "KRW", "₪": "ILS", "₫": "VND",
    "€": "EUR", "₱": "PHP", "₲": "PYG", "₴": "UAH", "₹": "INR",
}

POS_REGEX_PATTERNS = {
    "en": {
        "NP": r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+",
        "PP": r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+",
        "VP": r"<AUX>* <ADV>* <VERB>",
    }
}

RE_ACRONYM = re.compile(
    r"(?:^|(?<=\W))"
    r"(?:(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|[0-9]s?))|(?:[0-9](?:\-?[A-Z])+))"
    r"(?:$|(?=\W))",
    flags=re.UNICODE)
RE_EMAIL = re.compile(
    r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}"
    r"(?:$|(?=\b))",
    flags=re.IGNORECASE | re.UNICODE)
RE_PHONE = re.compile(
    # core components of a phone number
    r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})"
    # extensions, etc.
    r"(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))",
    flags=re.IGNORECASE)
RE_NUMBERS = re.compile(
    r"(?:^|(?<=[^\w,.]))[+–-]?"
    r"(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)"
    r"(?:$|(?=\b))")
RE_CURRENCY = re.compile(
    "({})+".format("|".join(re.escape(c) for c in CURRENCIES.keys())))
RE_LINEBREAK = re.compile(r"((\r\n)|[\n\v])+")
RE_NONBREAKING_SPACE = re.compile(r"(?!\n)\s+")
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
    flags=re.UNICODE | re.IGNORECASE)  # source: https://gist.github.com/dperini/729294
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
    flags=re.IGNORECASE)

# regexes for cleaning up crufty terms
RE_DANGLING_PARENS_TERM = re.compile(
    r"(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)", flags=re.UNICODE)
RE_LEAD_TAIL_CRUFT_TERM = re.compile(
    r"^([^\w(-] ?)+|([^\w).!?] ?)+$", flags=re.UNICODE)
RE_LEAD_HYPHEN_TERM = re.compile(
    r"^-([^\W\d_])", flags=re.UNICODE)
RE_NEG_DIGIT_TERM = re.compile(
    r"(-) (\d)", flags=re.UNICODE)
RE_WEIRD_HYPHEN_SPACE_TERM = re.compile(
    r"(?<=[^\W\d]) (-[^\W\d])", flags=re.UNICODE)
RE_WEIRD_APOSTR_SPACE_TERM = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE)


class PunctTranslateUnicode(object):
    """
    Class to make loading of unicode data into a punctuation translation mapping
    lazy, and only computed one time if at all. This speeds up package imports
    significantly.

    No, technically this isn't a constant, but I'm prepared to make an exception.
    """

    def __init__(self):
        self._data = None

    @property
    def data(self):
        if not self._data:
            self._data = dict.fromkeys(
                (i for i in compat.range_(sys.maxunicode)
                 if unicodedata.category(compat.chr_(i)).startswith("P")),
                " "
            )
        return self._data


PUNCT_TRANSLATE_UNICODE = PunctTranslateUnicode()
