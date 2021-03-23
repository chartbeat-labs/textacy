from __future__ import annotations

import re
from typing import Iterable, Pattern

from spacy.tokens import Doc, Span


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
