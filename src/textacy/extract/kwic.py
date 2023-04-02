"""
KWIC
----

:mod:`textacy.extract.kwic`: Extract keywords with their surrounding contexts from
a text document using regular expressions.
"""
from __future__ import annotations

import re
from typing import Iterable, Pattern

from spacy.tokens import Doc


def keyword_in_context(
    doc: Doc | str,
    keyword: str | Pattern,
    *,
    ignore_case: bool = True,
    window_width: int = 50,
    pad_context: bool = False,
) -> Iterable[tuple[str, str, str]]:
    """
    Search for ``keyword`` matches in ``doc`` via regular expression and yield matches
    along with ``window_width`` characters of context before and after occurrence.

    Args:
        doc: spaCy ``Doc`` or raw text in which to search for ``keyword``. If a ``Doc``,
            constituent text is grabbed via :attr:`spacy.tokens.Doc.text`. Note that
            spaCy annotations aren't used at all here, they're just a convenient
            owner of document text.
        keyword: String or regular expression pattern defining the keyword(s) to match.
            Typically, this is a single word or short phrase ("spam", "spam and eggs"),
            but to account for variations, use regex (``r"[Ss]pam (and|&) [Ee]ggs?"``),
            optionally compiled (``re.compile(r"[Ss]pam (and|&) [Ee]ggs?")``).
        ignore_case: If True, ignore letter case in ``keyword`` matching; otherwise,
            use case-sensitive matching. Note that this argument is only used if
            ``keyword`` is a string; for pre-compiled regular expressions,
            the ``re.IGNORECASE`` flag is left as-is.
        window_width: Number of characters on either side of ``keyword``
            to include as "context".
        pad_context: If True, pad pre- and post-context strings to ``window_width``
            chars in length; otherwise, us as many chars as are found in the text,
            up to the specified width.

    Yields:
        Next matching triple of (pre-context, keyword match, post-context).
    """
    text = doc.text if isinstance(doc, Doc) else doc
    if isinstance(keyword, str):
        flags = re.IGNORECASE if ignore_case is True else 0
        matches = re.finditer(keyword, text, flags=flags)
    else:
        matches = keyword.finditer(text)
    for match in matches:
        pre_context = text[max(0, match.start() - window_width) : match.start()]
        post_context = text[match.end() : match.end() + window_width]
        if pad_context is True:
            pre_context = pre_context.rjust(window_width)
            post_context = post_context.ljust(window_width)

        yield (pre_context, match.group(), post_context)
