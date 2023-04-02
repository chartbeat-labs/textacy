"""
Matches
-------

:mod:`textacy.extract.matches`: Extract matching spans from a document or sentence
using spaCy's built-in matcher or regular expressions.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Literal, Optional, Pattern, Union

from spacy.matcher import Matcher
from spacy.tokens import Span

from .. import constants, errors, types


def token_matches(
    doclike: types.DocLike,
    patterns: str | list[str] | list[dict[str, str]] | list[list[dict[str, str]]],
    *,
    on_match: Optional[Callable] = None,
) -> Iterable[Span]:
    """
    Extract ``Span`` s from a document or sentence matching one or more patterns
    of per-token attr:value pairs, with optional quantity qualifiers.

    Args:
        doclike
        patterns:
            One or multiple patterns to match against ``doclike``
            using a :class:`spacy.matcher.Matcher`.

            If list[dict] or list[list[dict]], each pattern is specified
            as attr: value pairs per token, with optional quantity qualifiers:

            - ``[{"POS": "NOUN"}]`` matches singular or plural nouns,
              like "friend" or "enemies"
            - ``[{"POS": "PREP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "?"}, {"POS": "NOUN", "OP": "+"}]``
              matches prepositional phrases, like "in the future" or "from the distant past"
            - ``[{"IS_DIGIT": True}, {"TAG": "NNS"}]`` matches numbered plural nouns,
              like "60 seconds" or "2 beers"
            - ``[{"POS": "PROPN", "OP": "+"}, {}]`` matches proper nouns and
              whatever word follows them, like "Burton DeWilde yaaasss"

            If str or list[str], each pattern is specified as one or more
            per-token patterns separated by whitespace where attribute, value,
            and optional quantity qualifiers are delimited by colons. Note that
            boolean and integer values have special syntax --- "bool(val)" and
            "int(val)", respectively --- and that wildcard tokens still need
            a colon between the (empty) attribute and value strings.

            - ``"POS:NOUN"`` matches singular or plural nouns
            - ``"POS:PREP POS:DET:? POS:ADJ:? POS:NOUN:+"`` matches prepositional phrases
            - ``"IS_DIGIT:bool(True) TAG:NNS"`` matches numbered plural nouns
            - ``"POS:PROPN:+ :"`` matches proper nouns and whatever word follows them

            Also note that these pattern strings don't support spaCy v2.1's
            "extended" pattern syntax; if you need such complex patterns, it's
            probably better to use a list[dict] or list[list[dict]], anyway.

        on_match: Callback function to act on matches.
            Takes the arguments ``matcher``, ``doclike``, ``i`` and ``matches``.

    Yields:
        Next matching ``Span`` in ``doclike``, in order of appearance

    Raises:
        TypeError
        ValueError

    See Also:
        - https://spacy.io/usage/rule-based-matching
        - https://spacy.io/api/matcher
    """  # noqa: E501
    if isinstance(patterns, str):
        patterns = [_make_pattern_from_string(patterns)]
    elif isinstance(patterns, (list, tuple)):
        if all(isinstance(item, str) for item in patterns):
            patterns = [_make_pattern_from_string(pattern) for pattern in patterns]  # type: ignore
        elif all(isinstance(item, dict) for item in patterns):
            patterns = [patterns]  # type: ignore
        elif all(isinstance(item, (list, tuple)) for item in patterns):
            pass  # already in the right format!
        else:
            raise TypeError(
                errors.type_invalid_msg(
                    "patterns",
                    type(patterns),
                    Union[
                        str, list[str], list[dict[str, str]], list[list[dict[str, str]]]
                    ],
                )
            )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "patterns",
                type(patterns),
                Union[str, list[str], list[dict[str, str]], list[list[dict[str, str]]]],
            )
        )
    matcher = Matcher(doclike.vocab)
    matcher.add("match", patterns, on_match=on_match)
    for match in matcher(doclike, as_spans=True):
        yield match


def _make_pattern_from_string(patstr: str) -> list[dict[str, str]]:
    pattern = []
    for tokpatstr in constants.RE_MATCHER_TOKPAT_DELIM.split(patstr):
        parts = tokpatstr.split(":")
        if 2 <= len(parts) <= 3:
            attr = parts[0]
            attr_val = parts[1]
            if attr and attr_val:
                # handle special bool and int attribute values
                special_val = constants.RE_MATCHER_SPECIAL_VAL.match(attr_val)
                if special_val:
                    attr_val = eval(special_val.group(0))
                tokpat = {attr: attr_val}
            # handle wildcard tokens
            else:
                tokpat = {}
            # handle quantifier ops
            try:
                op_val = parts[2]
                if op_val in constants.MATCHER_VALID_OPS:
                    tokpat["OP"] = op_val
                else:
                    raise ValueError(
                        errors.value_invalid_msg(
                            "op", op_val, constants.MATCHER_VALID_OPS
                        )
                    )
            except IndexError:
                pass
            pattern.append(tokpat)
        else:
            raise ValueError(
                f"pattern string '{patstr}' is invalid; "
                "each element in a pattern string must contain an attribute, "
                "a corresponding value, and an optional quantity qualifier, "
                "delimited by colons, like attr:value:op"
            )
    return pattern


def regex_matches(
    doclike: types.DocLike,
    pattern: str | Pattern,
    *,
    alignment_mode: Literal["strict", "contract", "expand"] = "strict",
) -> Iterable[Span]:
    """
    Extract ``Span`` s from a document or sentence whose full texts match against
    a regular expression ``pattern``.

    Args:
        doclike
        pattern: Valid regular expression against which to match document text,
            either as a string or compiled pattern object.
        alignment_mode: How character indices of regex matches snap to spaCy token
            boundaries. If "strict", only exact alignments are included (no snapping);
            if "contract", tokens completely within the character span are included;
            if "expand", tokens at least partially covered by the character span
            are included.

    Yields:
        Next matching ``Span``.
    """
    for match in re.finditer(pattern, doclike.text):
        start_char_idx, end_char_idx = match.span()
        span = doclike.char_span(
            start_char_idx, end_char_idx, alignment_mode=alignment_mode  # type: ignore
        )
        # Doc.char_span() returns None if character indices don’t map to a valid span
        if span is not None:
            yield span
