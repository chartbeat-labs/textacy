from __future__ import annotations

import operator
from typing import Any, Callable, Collection, Dict, List, Optional

import spacy
import cytoolz
from spacy.tokens import Doc, Span

from . import errors, types


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
    doclike: types.DocLike,
    *,
    by: str = "lemma_",  # Literal["lemma", "lemma_", "lower", "lower_", "norm", "norm_", "orth", "orth_"]
    weighting: str = "count",  # Literal["count", "freq", "binary"]
    **kwargs,
) -> Dict[int, int | float] | Dict[str, int | float]:
    """
    Transform a ``Doc`` or ``Span`` into a bag-of-words: the set of unique words therein
    mapped to their absolute, relative, or binary frequencies of occurrence.

    Args:
        doclike
        by: Attribute by which spaCy ``Token`` s are grouped before counting,
            as given by ``getattr(token, by)``.
            If "lemma", tokens are counted by their base form w/o inflectional suffixes;
            if "lower", by the lowercase form of the token text;
            if "norm", by the normalized form of the token text;
            if "orth", by the token text exactly as it appears in ``doc``.
            To output keys as strings, simply append an underscore to any of these;
            for example, "lemma_" creates a bag whose keys are token lemmas as strings.
        weighting: Type of weighting to assign to unique words given by ``by``.
            If "count", weights are the absolute number of occurrences (i.e. counts);
            if "freq", weights are counts normalized by the total token count,
            giving their relative frequency of occurrence;
            if "binary", weights are set equal to 1.
        **kwargs: Passed directly on to :func:`textacy.extract.words()`
            - filter_stops: If True, stop words are removed before counting.
            - filter_punct: If True, punctuation tokens are removed before counting.
            - filter_nums: If True, number-like tokens are removed before counting.

    Returns:
        Mapping of a unique word id or string (depending on the value of ``by``)
        to its absolute, relative, or binary frequency of occurrence
        (depending on the value of ``weighting``).

    Note:
        For "freq" weighting, the resulting set of frequencies won't (necessarily) sum
        to 1.0, since all tokens are used when normalizing counts but some (punctuation,
        stop words, etc.) may be filtered out of the bag afterwards.

    See Also:
        :func:`textacy.extract.words()`
    """
    from . import extract  # HACK: hide the import, ugh

    words = extract.words(doclike, **kwargs)
    bow = cytoolz.recipes.countby(operator.attrgetter(by), words)
    bow = _reweight_bag(weighting, bow, doclike)
    return bow


def to_bag_of_terms(
    doclike: types.DocLike,
    *,
    by: str = "lemma_",  # Literal["lemma_", "lemma", "lower_", "lower", "orth_", "orth"]
    weighting: str = "count",  # Literal["count", "freq", "binary"]
    ngs: Optional[int | Collection[int] | types.DocLikeToSpans] = None,
    ents: Optional[bool | types.DocLikeToSpans] = None,
    ncs: Optional[bool | types.DocLikeToSpans] = None,
    dedupe: bool = True,
) -> Dict[str, int] | Dict[str, float]:
    """
    Transform a ``Doc`` or ``Span`` into a bag-of-terms: the set of unique terms therein
    mapped to their absolute, relative, or binary frequencies of occurrence,
    where "terms" may be a combination of n-grams, entities, and/or noun chunks.

    Args:
        doclike
        by: Attribute by which spaCy ``Span`` s are grouped before counting,
            as given by ``getattr(token, by)``.
            If "lemma", tokens are counted by their base form w/o inflectional suffixes;
            if "lower", by the lowercase form of the token text;
            if "orth", by the token text exactly as it appears in ``doc``.
            To output keys as strings, simply append an underscore to any of these;
            for example, "lemma_" creates a bag whose keys are token lemmas as strings.
        weighting: Type of weighting to assign to unique terms given by ``by``.
            If "count", weights are the absolute number of occurrences (i.e. counts);
            if "freq", weights are counts normalized by the total token count,
            giving their relative frequency of occurrence;
            if "binary", weights are set equal to 1.
        ngs: N-gram terms to be extracted.
            If one or multiple ints, :func:`textacy.extract.ngrams(doclike, n=ngs)` is
            used to extract terms; if a callable, ``ngs(doclike)`` is used to extract
            terms; if None, no n-gram terms are extracted.
        ents: Entity terms to be extracted.
            If True, :func:`textacy.extract.entities(doclike)` is used to extract terms;
            if a callable, ``ents(doclike)`` is used to extract terms;
            if None, no entity terms are extracted.
        ncs: Noun chunk terms to be extracted.
            If True, :func:`textacy.extract.noun_chunks(doclike)` is used to extract
            terms; if a callable, ``ncs(doclike)`` is used to extract terms;
            if None, no noun chunk terms are extracted.
        dedupe: If True, deduplicate terms whose spans are extracted by multiple types
            (e.g. a span that is both an n-gram and an entity), as identified by
            identical (start, stop) indexes in ``doclike``; otherwise, don't.

    Returns:
        Mapping of a unique term id or string (depending on the value of ``by``)
        to its absolute, relative, or binary frequency of occurrence
        (depending on the value of ``weighting``).

    See Also:
        :func:`textacy.extract.terms()`
    """
    from . import extract  # HACK: hide the import, ugh

    terms = extract.terms(doclike, ngs=ngs, ents=ents, ncs=ncs, dedupe=dedupe)
    # spaCy made some awkward changes in the Span API, making it harder to get int ids
    # and adding inconsistencies with equivalent Token API
    # so, here are some hacks to spare users that annoyance
    if by.startswith("lower"):
        bot = cytoolz.recipes.countby(lambda span: span.text.lower(), terms)
    else:
        by_ = by if by.endswith("_") else f"{by}_"
        bot = cytoolz.recipes.countby(operator.attrgetter(by_), terms)
    # if needed, here we take the term strings and manually convert back to ints
    if not by.endswith("_"):
        ss = doclike.vocab.strings
        bot = {ss.add(term_str): weight for term_str, weight in bot.items()}
    bot = _reweight_bag(weighting, bot, doclike)
    return bot


def _reweight_bag(
    weighting: str, bag: Dict[Any, int], doclike: types.DocLike
) -> Dict[Any, int] | Dict[Any, float]:
    if weighting == "count":
        return bag
    elif weighting == "freq":
        n_tokens = len(doclike)
        return {term: weight / n_tokens for term, weight in bag.items()}
    elif weighting == "binary":
        return {term: 1 for term in bag.keys()}
    else:
        raise ValueError(
            errors.value_invalid_msg(
                "weighting", weighting, {"count", "freq", "binary"}
            )
        )


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
    "bag_of_terms": {"method": to_bag_of_terms},
}
