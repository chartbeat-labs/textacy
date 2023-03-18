"""
Acronyms
--------

:mod:`textacy.extract.acronyms`: Extract acronyms and their definitions from a document
or sentence through rule-based pattern-matching of the annotated tokens.
"""
from __future__ import annotations

import collections
from operator import itemgetter
from typing import Iterable, Optional

import numpy as np
from spacy.tokens import Span, Token

from .. import constants, types


def acronyms(doclike: types.DocLike) -> Iterable[Token]:
    """
    Extract tokens whose text is "acronym-like" from a document or sentence,
    in order of appearance.

    Args:
        doclike

    Yields:
        Next acronym-like ``Token``.
    """
    for tok in doclike:
        if is_acronym(tok.text):
            yield tok


def acronyms_and_definitions(
    doclike: types.DocLike,
    known_acro_defs: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Extract a collection of acronyms and their most likely definitions, if available,
    from a spacy-parsed doc. If multiple definitions are found for a given acronym,
    only the most frequently occurring definition is returned.

    Args:
        doclike
        known_acro_defs: If certain acronym/definition pairs
            are known, pass them in as {acronym (str): definition (str)};
            algorithm will not attempt to find new definitions

    Returns:
        Unique acronyms (keys) with matched definitions (values)

    References:
        Taghva, Kazem, and Jeff Gilbreth. "Recognizing acronyms and their definitions."
        International Journal on Document Analysis and Recognition 1.4 (1999): 191-198.
    """
    # process function arguments
    acro_defs: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    if not known_acro_defs:
        known_acronyms = set()
    else:
        for acro, def_ in known_acro_defs.items():
            acro_defs[acro] = [(def_, 1.0)]
        known_acronyms = set(acro_defs.keys())

    sents: Iterable[Span]
    if isinstance(doclike, Span):
        sents = [doclike]
    else:  # spacy.Doc
        sents = doclike.sents

    # iterate over sentences and their tokens
    for sent in sents:
        max_ind = len(sent) - 1

        for i, token in enumerate(sent):
            token_ = token.text
            if token_ in known_acronyms or is_acronym(token_) is False:
                continue

            # define definition search window(s)
            window_size = min(2 * len(token_), len(token_) + 5)
            windows = [
                sent[max(i - window_size, 0) : i],
                sent[min(i + 1, max_ind) : min(i + window_size + 1, max_ind)],
            ]
            # if candidate inside (X) or -X-, only look in pre-window
            if 0 < i < max_ind:
                adjacent_tokens = sent[i - 1].text + sent[i + 1].text
                if adjacent_tokens in {"()", "--", "––"}:
                    windows.pop()

            # iterate over possible windows
            # filtering for valid definition strings
            for window in windows:
                window_ = window.text
                # window text can't be all uppercase
                if window_.isupper():
                    continue
                # window can't contain separating punctuation
                if "!" in window_ or "?" in window_ or ":" in window_ or ";" in window_:
                    continue
                # acronym definition can't contain itself: no ouroboros!
                if token_ in window_:
                    continue
                # window must contain at least one character used in acronym
                if not any(char in window_ for char in token_):
                    continue
                definition, confidence = _get_acronym_definition(
                    token_, window, threshold=0.8
                )
                if definition:
                    acro_defs[token_].append((definition, confidence))

            if not acro_defs.get(token_):
                acro_defs[token_].append(("", 0.0))

    # vote by confidence score in the case of multiple definitions
    acro_defs_final: dict[str, str] = {}
    for acro, defs in acro_defs.items():
        if len(defs) == 1:
            acro_defs_final[acro] = defs[0][0]
        else:
            acro_defs_final[acro] = sorted(defs, key=itemgetter(1), reverse=True)[0][0]

    return acro_defs_final


def _get_acronym_definition(
    acronym: str,
    window: Span,
    threshold: float = 0.8,
) -> tuple[str, float]:
    """
    Identify most likely definition for an acronym given a list of tokens.

    Args:
        acronym: acronym for which definition is sought
        window: a span of tokens from which definition extraction will be attempted
        threshold: minimum "confidence" in definition required for acceptance;
            valid values in [0.0, 1.0]; higher value => stricter threshold

    Returns:
        Most likely definition for given acronym ('' if none found),
        along with the confidence assigned to it

    References:
        Taghva, Kazem, and Jeff Gilbreth. "Recognizing acronyms and their definitions."
        International Journal on Document Analysis and Recognition 1.4 (1999): 191-198.
    """

    def build_lcs_matrix(X, Y):
        m = len(X)
        n = len(Y)
        b = np.zeros((m, n), dtype=int)
        c = np.zeros((m, n), dtype=int)
        for i in range(0, m):
            for j in range(0, n):
                if X[i] == Y[j]:
                    c[i, j] = c[i - 1, j - 1] + 1
                    b[i, j] = 1
                elif c[i - 1, j] >= c[i, j - 1]:
                    c[i, j] = c[i - 1, j]
                else:
                    c[i, j] = c[i, j - 1]
        return c, b

    def parse_lcs_matrix(b, start_i, start_j, lcs_length, stack, vectors):
        m = b.shape[0]
        n = b.shape[1]
        for i in range(start_i, m):
            for j in range(start_j, n):
                if b[i, j] == 1:
                    s = (i, j)
                    stack.append(s)
                    if lcs_length == 1:
                        vec = [np.NaN] * n
                        for k, l in stack:
                            vec[l] = k
                        vectors.append(vec)
                    else:
                        parse_lcs_matrix(
                            b, i + 1, j + 1, lcs_length - 1, stack, vectors
                        )
                    stack = []
        return vectors

    def vector_values(v, types):
        vv = {}
        first = v.index(int(np.nanmin(v)))
        last = v.index(int(np.nanmax(v)))
        vv["size"] = (last - first) + 1
        vv["distance"] = len(v) - last
        vv["stop_count"] = 0
        vv["misses"] = 0
        for i in range(first, last + 1):
            if v[i] >= 0 and types[i] == "s":
                vv["stop_count"] += 1
            elif v[i] is None and types[i] not in ["s", "h"]:
                vv["misses"] += 1
        return vv

    def compare_vectors(A, B, types):
        vv_A = vector_values(A, types)
        vv_B = vector_values(B, types)
        # no one-letter matches, sorryboutit
        if vv_A["size"] == 1:
            return B
        elif vv_B["size"] == 1:
            return A
        if vv_A["misses"] > vv_B["misses"]:
            return B
        elif vv_A["misses"] < vv_B["misses"]:
            return A
        if vv_A["stop_count"] > vv_B["stop_count"]:
            return B
        if vv_A["stop_count"] < vv_B["stop_count"]:
            return A
        if vv_A["distance"] > vv_B["distance"]:
            return B
        elif vv_A["distance"] < vv_B["distance"]:
            return A
        if vv_A["size"] > vv_B["size"]:
            return B
        elif vv_A["size"] < vv_B["size"]:
            return A
        return A

    # get definition window's leading characters and word types
    def_leads = []
    def_types = []
    for tok in window:
        tok_text = tok.text
        if tok.is_stop:
            def_leads.append(tok_text[0])
            def_types.append("s")
        elif is_acronym(tok_text):
            def_leads.append(tok_text[0])
            def_types.append("a")
        elif "-" in tok_text and not tok_text.startswith("-"):
            tok_split = [t[0] for t in tok_text.split("-") if t]
            def_leads.extend(tok_split)
            def_types.extend("H" if i == 0 else "h" for i in range(len(tok_split)))
        else:
            def_leads.append(tok_text[0])
            def_types.append("w")
    def_leads = "".join(def_leads).lower()
    def_types = "".join(def_types)

    # extract alphanumeric characters from acronym
    acr_leads = "".join(c for c in acronym if c.isalnum())
    # handle special cases of '&' and trailing 's'
    acr_leads = acr_leads.replace("&", "a")
    if acr_leads.endswith("s"):
        # bail out if it's only a 2-letter acronym to start with, e.g. 'Is'
        if len(acr_leads) == 2:
            return ("", 0)
        acr_leads = acr_leads[:-1]
    acr_leads = acr_leads.lower()

    c, b = build_lcs_matrix(acr_leads, def_leads)

    # 4.4.1
    lcs_length = c[c.shape[0] - 1, c.shape[1] - 1]
    confidence = lcs_length / len(acronym)
    if confidence < threshold:
        return ("", confidence)

    vecs = parse_lcs_matrix(b, 0, 0, lcs_length, [], [])
    # first letter of acronym must be present
    vecs = [vec for vec in vecs if 0 in vec]
    if not vecs:
        return ("", confidence)

    best_vec = vecs[0]
    for vec in vecs[1:]:
        best_vec = compare_vectors(best_vec, vec, def_types)

    first = best_vec.index(int(np.nanmin(best_vec)))
    last = best_vec.index(int(np.nanmax(best_vec)))

    definition = window[first : last + 1].text
    if len(definition.split()) == 1:
        return ("", confidence)

    return (definition, confidence)


def is_acronym(token: str, exclude: Optional[set[str]] = None) -> bool:
    """
    Pass single token as a string, return True/False if is/is not valid acronym.

    Args:
        token: Single word to check for acronym-ness
        exclude: If technically valid but not actual acronyms are known in advance,
            pass them in as a set of strings; matching tokens will return False.

    Returns:
        Whether or not ``token`` is an acronym.
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
