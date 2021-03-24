from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Pattern, Union

from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from .. import constants, errors


def token_matches(
    doc: Doc,
    patterns: str | List[str] | List[Dict[str, str]] | List[List[Dict[str, str]]],
    *,
    on_match: Optional[Callable] = None,
) -> Iterable[Span]:
    """
    Extract ``Span`` s from a ``Doc`` matching one or more patterns
    of per-token attr:value pairs, with optional quantity qualifiers.

    Args:
        doc
        patterns:
            One or multiple patterns to match against ``doc``
            using a :class:`spacy.matcher.Matcher`.

            If List[dict] or List[List[dict]], each pattern is specified
            as attr: value pairs per token, with optional quantity qualifiers:

            * ``[{"POS": "NOUN"}]`` matches singular or plural nouns,
              like "friend" or "enemies"
            * ``[{"POS": "PREP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "?"}, {"POS": "NOUN", "OP": "+"}]``
              matches prepositional phrases, like "in the future" or "from the distant past"
            * ``[{"IS_DIGIT": True}, {"TAG": "NNS"}]`` matches numbered plural nouns,
              like "60 seconds" or "2 beers"
            * ``[{"POS": "PROPN", "OP": "+"}, {}]`` matches proper nouns and
              whatever word follows them, like "Burton DeWilde yaaasss"

            If str or List[str], each pattern is specified as one or more
            per-token patterns separated by whitespace where attribute, value,
            and optional quantity qualifiers are delimited by colons. Note that
            boolean and integer values have special syntax --- "bool(val)" and
            "int(val)", respectively --- and that wildcard tokens still need
            a colon between the (empty) attribute and value strings.

            * ``"POS:NOUN"`` matches singular or plural nouns
            * ``"POS:PREP POS:DET:? POS:ADJ:? POS:NOUN:+"`` matches prepositional phrases
            * ``"IS_DIGIT:bool(True) TAG:NNS"`` matches numbered plural nouns
            * ``"POS:PROPN:+ :"`` matches proper nouns and whatever word follows them

            Also note that these pattern strings don't support spaCy v2.1's
            "extended" pattern syntax; if you need such complex patterns, it's
            probably better to use a List[dict] or List[List[dict]], anyway.

        on_match: Callback function to act on matches.
            Takes the arguments ``matcher``, ``doc``, ``i`` and ``matches``.

    Yields:
        Next matching ``Span`` in ``doc``, in order of appearance

    Raises:
        TypeError
        ValueError

    See Also:
        * https://spacy.io/usage/rule-based-matching
        * https://spacy.io/api/matcher
    """  # noqa: E501
    if isinstance(patterns, str):
        patterns = [_make_pattern_from_string(patterns)]
    elif isinstance(patterns, (list, tuple)):
        if all(isinstance(item, str) for item in patterns):
            patterns = [_make_pattern_from_string(pattern) for pattern in patterns]
        elif all(isinstance(item, dict) for item in patterns):
            patterns = [patterns]
        elif all(isinstance(item, (list, tuple)) for item in patterns):
            pass  # already in the right format!
        else:
            raise TypeError(
                errors.type_invalid_msg(
                    "patterns",
                    type(patterns),
                    Union[
                        str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]
                    ],
                )
            )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "patterns",
                type(patterns),
                Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
            )
        )
    matcher = Matcher(doc.vocab)
    matcher.add("match", patterns, on_match=on_match)
    for _, start, end in matcher(doc):
        yield doc[start:end]


def _make_pattern_from_string(patstr: str) -> List[Dict[str, str]]:
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
    doclike: Doc | Span,
    pattern: str | Pattern,
    *,
    expand: bool = False,
) -> Iterable[Span]:
    """
    Extract ``Span`` s from a Doc-like object whose full texts match against
    a regular expression ``pattern``.

    Args:
        doclike
        pattern: Valid regular expression against which to match document text,
            either as a string or compiled pattern object.
        expand: If True, automatically expand matches to include full overlapping tokens
            in the case that matches don't exactly align with spaCy's tokenization
            (i.e. one or both edges fall somewhere mid-token). Otherwise, only match
            when edges are aligned with token boundaries.

    Yields:
        Next matching ``Span``.
    """
    if expand is True:
        char_to_tok_idxs = {
            char_idx: tok.i
            for tok in doclike
            for char_idx in range(tok.idx, tok.idx + len(tok.text))
        }
    for match in re.finditer(pattern, doclike.text):
        start_char_idx, end_char_idx = match.span()
        span = doclike.char_span(start_char_idx, end_char_idx)
        # .char_span() returns None if character indices don’t map to a valid span
        if span is not None:
            yield span
        # in case of matches that don't align with token boundaries, we can expand out
        # to capture the corresponding start/end tokens
        elif expand is True:
            start_tok_idx = char_to_tok_idxs.get(start_char_idx)
            end_tok_idx = char_to_tok_idxs.get(end_char_idx)
            if start_tok_idx is not None and end_tok_idx is not None:
                yield doclike[start_tok_idx : end_tok_idx + 1]
