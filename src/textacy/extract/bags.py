from __future__ import annotations

import operator
from typing import Any, Collection, Literal, Optional, Union

import cytoolz

from .. import errors, types
from . import basics


WeightingType = Literal["count", "freq", "binary"]
SpanGroupByType = Literal["lemma", "lemma_", "lower", "lower_", "orth", "orth_"]
TokenGroupByType = Union[SpanGroupByType, Literal["norm", "norm_"]]


def to_bag_of_words(
    doclike: types.DocLike,
    *,
    by: TokenGroupByType = "lemma_",
    weighting: WeightingType = "count",
    **kwargs,
) -> dict[int, int | float] | dict[str, int | float]:
    """
    Transform a ``Doc`` or ``Span`` into a bag-of-words: the set of unique words therein
    mapped to their absolute, relative, or binary frequencies of occurrence.

    Args:
        doclike
        by: Attribute by which spaCy ``Token`` s are grouped before counting,
            as given by ``getattr(token, by)``.
            If "lemma", tokens are grouped by their base form w/o inflectional suffixes;
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
    words = basics.words(doclike, **kwargs)
    bow = cytoolz.recipes.countby(operator.attrgetter(by), words)
    bow = _reweight_bag(weighting, bow, doclike)
    return bow


def to_bag_of_terms(
    doclike: types.DocLike,
    *,
    by: SpanGroupByType = "lemma_",
    weighting: WeightingType = "count",
    ngs: Optional[int | Collection[int] | types.DocLikeToSpans] = None,
    ents: Optional[bool | types.DocLikeToSpans] = None,
    ncs: Optional[bool | types.DocLikeToSpans] = None,
    dedupe: bool = True,
) -> dict[str, int] | dict[str, float]:
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
    terms = basics.terms(doclike, ngs=ngs, ents=ents, ncs=ncs, dedupe=dedupe)
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
    weighting: WeightingType, bag: dict[Any, int], doclike: types.DocLike
) -> dict[Any, int] | dict[Any, float]:
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
