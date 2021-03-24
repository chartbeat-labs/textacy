"""
:mod:`textacy.text_utils`: Set of small utility functions that take text strings as input.
"""
import logging
import re
from typing import Iterable, Optional, Set

from . import constants

LOGGER = logging.getLogger(__name__)


def clean_terms(terms: Iterable[str]) -> Iterable[str]:
    """
    Clean up a sequence of single- or multi-word strings: strip leading/trailing
    junk chars, handle dangling parens and odd hyphenation, etc.

    Args:
        terms: Sequence of terms such as "presidency", "epic failure",
            or "George W. Bush" that may be _unclean_ for whatever reason.

    Yields:
        Next term in `terms` but with the cruft cleaned up, excluding terms
        that were _entirely_ cruft

    Warning:
        Terms with (intentionally) unusual punctuation may get "cleaned"
        into a form that changes or obscures the original meaning of the term.
    """
    # get rid of leading/trailing junk characters
    terms = (constants.RE_LEAD_TAIL_CRUFT_TERM.sub("", term) for term in terms)
    terms = (constants.RE_LEAD_HYPHEN_TERM.sub(r"\1", term) for term in terms)
    # handle dangling/backwards parens, don't allow '(' or ')' to appear alone
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
