from __future__ import annotations

from typing import Any, Dict, List

import spacy
from spacy.tokens import Doc

from . import errors


def get_preview(doc: Doc) -> str:
    """
    Get a short preview of the ``Doc``, including the number of tokens
    and an initial snippet.
    """
    snippet = doc.text[:50].replace("\n", " ")
    if len(snippet) == 50:
        snippet = snippet[:47] + "..."
    return f'Doc({len(doc)} tokens: "{snippet}")'


def get_meta(doc: Doc) -> dict:
    """Get custom metadata added to ``Doc``."""
    return doc.user_data.get("textacy", {}).get("meta", {})


def set_meta(doc: Doc, value: dict) -> None:
    """Add custom metadata to ``Doc``."""
    if not isinstance(value, dict):
        raise TypeError(errors.type_invalid_msg("value", type(value), Dict))
    try:
        doc.user_data["textacy"]["meta"] = value
    except KeyError:
        # TODO: confirm that this is the same. it is, right??
        doc.user_data["textacy"] = {"meta": value}
        # doc.user_data["textacy"] = {}
        # doc.user_data["textacy"]["meta"] = value


def to_tokenized_text(doc: Doc) -> List[List[str]]:
    """
    Transform ``doc`` into an ordered, nested list of token-texts for each sentence.

    Args:
        doc

    Returns:
        A list of tokens' texts for each sentence in ``doc``.

    Note:
        If ``doc`` hasn't been segmented into sentences, the entire document
        is treated as a single sentence.
    """
    if doc.has_annotation("SENT_START"):
        return [[token.text for token in sent] for sent in doc.sents]
    else:
        return [[token.text for token in doc]]


def to_bag_of_words(
    doc: Doc,
    *,
    by: str = "lemma",  # Literal["lemma", "lower", "norm", "orth"]
    weighting: str = "count",  # Literal["count", "freq", "binary"]
    as_strings: bool = True,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
) -> Dict[int, int | float] | Dict[str, int | float]:
    """
    Transform ``doc`` into a bag-of-words: the set of unique words in ``doc``
    mapped to their absolute, relative, or binary frequencies of occurrence.

    Args:
        doc
        by: Token attribute by which tokens are grouped before counting.
            If "lemma", tokens are counted by their base form w/o inflectional suffixes;
            if "lower", by the lowercase form of the token text;
            if "norm", by the normalized form of the token text;
            if "orth", by the token text exactly as it appears in ``doc``.
        weighting: Type of weighting to assign to unique words given by ``by``.
            If "count", weights are the absolute number of occurrences (i.e. counts);
            if "freq", weights are counts normalized by the total token count,
            giving their relative frequency of occurrence;
            if "binary", weights are set equal to 1.
        as_strings: If True, return words as strings; otherwise, return words as
            the unique integer ids specified by ``getattr(Token, by)``.
        filter_stops: If True, stop words are removed after counting.
        filter_punct: If True, punctuation tokens are removed after counting.
        filter_nums: If True, number-like tokens are removed after counting.

    Returns:
        Mapping of a unique word id or string (depending on the value of ``as_strings``)
        to its absolute, relative, or binary frequency of occurrence
        (depending on the value of ``weighting``).

    Note:
        For "freq" weighting, the resulting set of frequencies won't (necessarily) sum
        to 1.0, since all tokens are used when normalizing counts but some (punctuation,
        stop words, etc.) may be filtered out of the bag afterwards.
    """
    attr_id = getattr(spacy.attrs, by.upper())
    attr_weights = doc.count_by(attr_id)
    if weighting == "freq":
        n_tokens = len(doc)
        attr_weights = {
            attr_id: weight / n_tokens for attr_id, weight in attr_weights.items()
        }
    elif weighting == "binary":
        attr_weights = {attr_id: 1 for attr_id in attr_weights.keys()}

    bow = {}
    vocab = doc.vocab
    for attr_id, weight in attr_weights.items():
        lex = vocab[attr_id]
        if not (
            (filter_stops and lex.is_stop) or
            (filter_punct and lex.is_punct) or
            (filter_nums and lex.is_digit) or
            lex.is_space
        ):
            if as_strings is True:
                bow[lex.text] = weight
            else:
                bow[attr_id] = weight

    return bow


def get_doc_extensions() -> Dict[str, Dict[str, Any]]:
    """
    Get textacy's custom property and method doc extensions
    that can be set on or removed from the global :class:`spacy.tokens.Doc`.
    """
    return _DOC_EXTENSIONS


def set_doc_extensions():
    """
    Set textacy's custom property and method doc extensions
    on the global :class:`spacy.tokens.Doc`.
    """
    for name, kwargs in get_doc_extensions().items():
        if not Doc.has_extension(name):
            Doc.set_extension(name, **kwargs)


def remove_doc_extensions():
    """
    Remove textacy's custom property and method doc extensions
    from the global :class:`spacy.tokens.Doc`.
    """
    for name in get_doc_extensions().keys():
        _ = Doc.remove_extension(name)


_DOC_EXTENSIONS: Dict[str, Dict[str, Any]] = {
    # property extensions
    "preview": {"getter": get_preview},
    "meta": {"getter": get_meta, "setter": set_meta},
    # method extensions
    "tokenized_text": {"method": to_tokenized_text},
    "bag_of_words": {"method": to_bag_of_words},
}
