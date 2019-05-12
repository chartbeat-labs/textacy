"""
Text Utils
----------

Set of small utility functions that take text strings as input.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re

from . import constants

LOGGER = logging.getLogger(__name__)


def is_acronym(token, exclude=None):
    """
    Pass single token as a string, return True/False if is/is not valid acronym.

    Args:
        token (str): single word to check for acronym-ness
        exclude (Set[str]): if technically valid but not actually good acronyms
            are known in advance, pass them in as a set of strings; matching
            tokens will return False

    Returns:
        bool
    """
    # exclude certain valid acronyms from consideration
    if exclude and token in exclude:
        return False
    # don't allow empty strings
    if not token:
        return False
    # don't allow spaces
    if " " in token:
        return False
    # 2-character acronyms can't have lower-case letters
    if len(token) == 2 and not token.isupper():
        return False
    # acronyms can't be all digits
    if token.isdigit():
        return False
    # acronyms must have at least one upper-case letter or start/end with a digit
    if not any(char.isupper() for char in token) and not (
        token[0].isdigit() or token[-1].isdigit()
    ):
        return False
    # acronyms must have between 2 and 10 alphanumeric characters
    if not 2 <= sum(1 for char in token if char.isalnum()) <= 10:
        return False
    # only certain combinations of letters, digits, and '&/.-' allowed
    if not constants.RE_ACRONYM.match(token):
        return False
    return True


def keyword_in_context(
    text, keyword, ignore_case=True, window_width=50, print_only=True
):
    """
    Search for ``keyword`` in ``text`` via regular expression, return or print strings
    spanning ``window_width`` characters before and after each occurrence of keyword.

    Args:
        text (str): text in which to search for ``keyword``
        keyword (str): technically, any valid regular expression string should work,
            but usually this is a single word or short phrase: "spam", "spam and eggs";
            to account for variations, use regex: "[Ss]pam (and|&) [Ee]ggs?"

            N.B. If keyword contains special characters, be sure to escape them!!!
        ignore_case (bool): if True, ignore letter case in `keyword` matching
        window_width (int): number of characters on either side of
            `keyword` to include as "context"
        print_only (bool): if True, print out all results with nice
            formatting; if False, return all (pre, kw, post) matches as generator
            of raw strings

    Returns:
        generator(Tuple[str, str, str]), or None
    """
    flags = re.IGNORECASE if ignore_case is True else 0
    if print_only is True:
        for match in re.finditer(keyword, text, flags=flags):
            line = "{pre} {kw} {post}".format(
                pre=text[max(0, match.start() - window_width) : match.start()].rjust(
                    window_width
                ),
                kw=match.group(),
                post=text[match.end() : match.end() + window_width].ljust(window_width),
            )
            print(line)
    else:
        return (
            (
                text[max(0, match.start() - window_width) : match.start()],
                match.group(),
                text[match.end() : match.end() + window_width],
            )
            for match in re.finditer(keyword, text, flags=flags)
        )


KWIC = keyword_in_context
"""Alias of :func:`keyword_in_context <textacy.text_utils.keyword_in_context>`."""


def clean_terms(terms):
    """
    Clean up a sequence of single- or multi-word strings: strip leading/trailing
    junk chars, handle dangling parens and odd hyphenation, etc.

    Args:
        terms (Iterable[str]): sequence of terms such as "presidency", "epic failure",
            or "George W. Bush" that may be _unclean_ for whatever reason

    Yields:
        str: next term in `terms` but with the cruft cleaned up, excluding terms
        that were _entirely_ cruft

    Warning:
        Terms with (intentionally) unusual punctuation may get "cleaned"
        into a form that changes or obscures the original meaning of the term.
    """
    # get rid of leading/trailing junk characters
    terms = (constants.RE_LEAD_TAIL_CRUFT_TERM.sub("", term) for term in terms)
    terms = (constants.RE_LEAD_HYPHEN_TERM.sub(r"\1", term) for term in terms)
    # handle dangling/backwards parens, don't allow '(' or ')' to appear without the other
    terms = (
        ""
        if term.count(")") != term.count("(") or term.find(")") < term.find("(")
        else term
        if "(" not in term
        else constants.RE_DANGLING_PARENS_TERM.sub(r"\1\2\3", term)
        for term in terms
    )
    # handle oddly separated hyphenated words
    terms = (
        term
        if "-" not in term
        else constants.RE_NEG_DIGIT_TERM.sub(
            r"\1\2", constants.RE_WEIRD_HYPHEN_SPACE_TERM.sub(r"\1", term)
        )
        for term in terms
    )
    # handle oddly separated apostrophe'd words
    terms = (
        constants.RE_WEIRD_APOSTR_SPACE_TERM.sub(r"\1\2", term) if "'" in term else term
        for term in terms
    )
    # normalize whitespace
    terms = (constants.RE_NONBREAKING_SPACE.sub(" ", term).strip() for term in terms)
    for term in terms:
        if re.search(r"\w", term):
            yield term
