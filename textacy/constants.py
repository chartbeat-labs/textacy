# -*- coding: utf-8 -*-
"""
Collection of regular expressions and other (small, generally useful) constants.
"""
from __future__ import unicode_literals

import re
import sys
import unicodedata

from . import compat


NUMERIC_NE_TYPES = {'ORDINAL', 'CARDINAL', 'MONEY', 'QUANTITY', 'PERCENT', 'TIME', 'DATE'}
SUBJ_DEPS = {'agent', 'csubj', 'csubjpass', 'expl', 'nsubj', 'nsubjpass'}
OBJ_DEPS = {'attr', 'dobj', 'dative', 'oprd'}
AUX_DEPS = {'aux', 'auxpass', 'neg'}

REPORTING_VERBS = {'according', 'accuse', 'acknowledge', 'add', 'admit', 'agree',
                   'allege', 'announce', 'argue', 'ask', 'assert', 'believe', 'blame',
                   'charge', 'cite', 'claim', 'complain', 'concede', 'conclude',
                   'confirm', 'contend', 'criticize', 'declare', 'decline', 'deny',
                   'describe', 'disagree', 'disclose', 'estimate', 'explain', 'fear',
                   'hope', 'insist', 'maintain', 'mention', 'note', 'observe', 'order',
                   'predict', 'promise', 'recall', 'recommend', 'reply', 'report', 'say',
                   'state', 'stress', 'suggest', 'tell', 'testify', 'think', 'urge', 'warn',
                   'worry', 'write'}

CURRENCIES = {'$': 'USD', 'zł': 'PLN', '£': 'GBP', '¥': 'JPY', '฿': 'THB',
              '₡': 'CRC', '₦': 'NGN', '₩': 'KRW', '₪': 'ILS', '₫': 'VND',
              '€': 'EUR', '₱': 'PHP', '₲': 'PYG', '₴': 'UAH', '₹': 'INR'}

POS_REGEX_PATTERNS = {
    'en': {'NP': r'<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+',
           'PP': r'<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+',
           'VP': r'<AUX>* <ADV>* <VERB>'}
    }

PUNCT_TRANSLATE_UNICODE = dict.fromkeys(
    (i for i in compat.range_(sys.maxunicode)
     if unicodedata.category(compat.chr_(i)).startswith('P')),
    u' ')

ACRONYM_REGEX = re.compile(r"(?:^|(?<=\W))(?:(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|[0-9]s?))|(?:[0-9](?:\-?[A-Z])+))(?:$|(?=\W))", flags=re.UNICODE)
EMAIL_REGEX = re.compile(r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}(?:$|(?=\b))", flags=re.IGNORECASE | re.UNICODE)
PHONE_REGEX = re.compile(r'(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))')
NUMBERS_REGEX = re.compile(r'(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))')
CURRENCY_REGEX = re.compile('({})+'.format('|'.join(re.escape(c) for c in CURRENCIES.keys())))
LINEBREAK_REGEX = re.compile(r'((\r\n)|[\n\v])+')
NONBREAKING_SPACE_REGEX = re.compile(r'(?!\n)\s+')
URL_REGEX = re.compile(
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
SHORT_URL_REGEX = re.compile(
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
DANGLING_PARENS_TERM_RE = re.compile(
    r'(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)', flags=re.UNICODE)
LEAD_TAIL_CRUFT_TERM_RE = re.compile(
    r'^([^\w(-] ?)+|([^\w).!?] ?)+$', flags=re.UNICODE)
LEAD_HYPHEN_TERM_RE = re.compile(
    r'^-([^\W\d_])', flags=re.UNICODE)
NEG_DIGIT_TERM_RE = re.compile(
    r'(-) (\d)', flags=re.UNICODE)
WEIRD_HYPHEN_SPACE_TERM_RE = re.compile(
    r'(?<=[^\W\d]) (-[^\W\d])', flags=re.UNICODE)
WEIRD_APOSTR_SPACE_TERM_RE = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE)
