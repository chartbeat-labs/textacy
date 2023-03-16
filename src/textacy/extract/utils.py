"""
Utils
-----

:mod:`textacy.extract.utils`: Functions for working with extraction results.
"""
from __future__ import annotations

import itertools
import operator
import re
from typing import Callable, Collection, Iterable, Optional

from cytoolz import itertoolz
from spacy.tokens import Doc, Token

from .. import constants, errors, types, utils
from . import matches


def terms_to_strings(
    terms: Iterable[types.SpanLike],
    by: str | Callable[[types.SpanLike], str],
) -> Iterable[str]:
    """
    Transform a sequence of terms as spaCy ``Token`` s or ``Span`` s into strings.

    Args:
        terms
        by: Method by which terms are transformed into strings.
            If "orth", terms are represented by their text exactly as written;
            if "lower", by the lowercased form of their text;
            if "lemma", by their base form w/o inflectional suffixes;
            if a callable, must accept a ``Token`` or ``Span`` and return a string.

    Yields:
        Next term in ``terms``, as a string.
    """
    terms_: Iterable[str]
    if by == "lower":
        terms_ = (term.text.lower() for term in terms)
    elif by in ("lemma", "orth"):
        by_ = operator.attrgetter(f"{by}_")
        terms_ = (by_(term) for term in terms)
    elif callable(by):
        terms_ = (by(term) for term in terms)
    else:
        raise ValueError(
            errors.value_invalid_msg("by", by, {"orth", "lower", "lemma", Callable})
        )
    for term in terms_:
        yield term


def clean_term_strings(terms: Iterable[str]) -> Iterable[str]:
    """
    Clean up a sequence of single- or multi-word terms as strings:
    strip leading/trailing junk chars, handle dangling parens and odd hyphenation,
    and normalize whitespace.

    Args:
        terms: Sequence of terms such as "environment", "plastic pollution",
            or "fossil fuel industry" that may be _unclean_ for whatever reason.

    Yields:
        Next term in ``terms`` but with the cruft cleaned up, excluding terms
        that were entirely cruft

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


def aggregate_term_variants(
    terms: set[str],
    *,
    acro_defs: Optional[dict[str, str]] = None,
    fuzzy_dedupe: bool = True,
) -> list[set[str]]:
    """
    Take a set of unique terms and aggregate terms that are symbolic, lexical,
    and ordering variants of each other, as well as acronyms and fuzzy string matches.

    Args:
        terms: Set of unique terms with potential duplicates
        acro_defs: If not None, terms that are acronyms will be aggregated
            with their definitions and terms that are definitions will be aggregated
            with their acronyms
        fuzzy_dedupe: If True, fuzzy string matching will be used
            to aggregate similar terms of a sufficient length

    Returns:
        Each item is a set of aggregated terms.

    Notes:
        Partly inspired by aggregation of variants discussed in
        Park, Youngja, Roy J. Byrd, and Branimir K. Boguraev.
        "Automatic glossary extraction: beyond terminology identification."
        Proceedings of the 19th international conference on Computational linguistics-
        Volume 1. Association for Computational Linguistics, 2002.
    """
    from .. import similarity  # ugh, hide import here

    agg_terms = []
    seen_terms: set[str] = set()
    for term in sorted(terms, key=len, reverse=True):
        if term in seen_terms:
            continue
        variants = set([term])
        seen_terms.add(term)

        # symbolic variations
        if "-" in term:
            variant = term.replace("-", " ").strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)
        if "/" in term:
            variant = term.replace("/", " ").strip()
            if variant in terms.difference(seen_terms):
                variants.add(variant)
                seen_terms.add(variant)

        # lexical variations
        term_words = term.split()
        # last_word = term_words[-1]
        # # assume last word is a noun
        # last_word_lemmatized = lemmatizer.lemmatize(last_word, 'n')
        # # if the same, either already a lemmatized noun OR a verb; try verb
        # if last_word_lemmatized == last_word:
        #     last_word_lemmatized = lemmatizer.lemmatize(last_word, 'v')
        # # if at least we have a new term... add it
        # if last_word_lemmatized != last_word:
        #     term_lemmatized = ' '.join(term_words[:-1] + [last_word_lemmatized])
        #     if term_lemmatized in terms.difference(seen_terms):
        #         variants.add(term_lemmatized)
        #         seen_terms.add(term_lemmatized)

        # if term is an acronym, add its definition
        # if term is a definition, add its acronym
        if acro_defs:
            for acro, def_ in acro_defs.items():
                if acro.lower() == term.lower():
                    variants.add(def_.lower())
                    seen_terms.add(def_.lower())
                    break
                elif def_.lower() == term.lower():
                    variants.add(acro.lower())
                    seen_terms.add(acro.lower())
                    break

        # if 3+ -word term differs by one word at the start or the end
        # of a longer phrase, aggregate
        if len(term_words) > 2:
            term_minus_first_word = " ".join(term_words[1:])
            term_minus_last_word = " ".join(term_words[:-1])
            if term_minus_first_word in terms.difference(seen_terms):
                variants.add(term_minus_first_word)
                seen_terms.add(term_minus_first_word)
            if term_minus_last_word in terms.difference(seen_terms):
                variants.add(term_minus_last_word)
                seen_terms.add(term_minus_last_word)
            # check for "X of Y" <=> "Y X" term variants
            if " of " in term:
                split_term = term.split(" of ")
                variant = split_term[1] + " " + split_term[0]
                if variant in terms.difference(seen_terms):
                    variants.add(variant)
                    seen_terms.add(variant)

        # intense de-duping for sufficiently long terms
        if fuzzy_dedupe is True and len(term) >= 13:
            for other_term in sorted(
                terms.difference(seen_terms), key=len, reverse=True
            ):
                if len(other_term) < 13:
                    break
                tsr = similarity.token_sort_ratio(term, other_term)
                if tsr > 0.93:
                    variants.add(other_term)
                    seen_terms.add(other_term)
                    break

        agg_terms.append(variants)

    return agg_terms


def get_longest_subsequence_candidates(
    doc: Doc,
    match_func: Callable[[Token], bool],
) -> Iterable[tuple[Token, ...]]:
    """
    Get candidate keyterms from ``doc``, where candidates are longest consecutive
    subsequences of tokens for which all ``match_func(token)`` is True.

    Args:
        doc
        match_func: Function applied sequentially to each ``Token`` in ``doc``
            that returns True for matching ("good") tokens, False otherwise.

    Yields:
        Next longest consecutive subsequence candidate, as a tuple of constituent tokens.
    """
    for key, words_grp in itertools.groupby(doc, key=match_func):
        if key is True:
            yield tuple(words_grp)


def get_ngram_candidates(
    doc: Doc,
    ns: int | Collection[int],
    *,
    include_pos: Optional[str | Collection[str]] = ("NOUN", "PROPN", "ADJ"),
) -> Iterable[tuple[Token, ...]]:
    """
    Get candidate keyterms from ``doc``, where candidates are n-length sequences
    of tokens (for all n in ``ns``) that don't start/end with a stop word or
    contain punctuation tokens, and whose constituent tokens are filtered by POS tag.

    Args:
        doc
        ns: One or more n values for which to generate n-grams. For example,
            ``2`` gets bigrams; ``(2, 3)`` gets bigrams and trigrams.
        include_pos: One or more POS tags with which to filter ngrams.
            If None, include tokens of all POS tags.

    Yields:
        Next ngram candidate, as a tuple of constituent Tokens.

    See Also:
        :func:`textacy.extract.ngrams()`
    """
    ns_: tuple[int, ...] = utils.to_tuple(ns)
    ngrams = itertoolz.concat(itertoolz.sliding_window(n, doc) for n in ns_)
    ngrams = (
        ngram
        for ngram in ngrams
        if not (ngram[0].is_stop or ngram[-1].is_stop)
        and not any(word.is_punct or word.is_space for word in ngram)
    )
    if include_pos:
        include_pos_: set[str] = utils.to_set(include_pos)
        ngrams = (
            ngram
            for ngram in ngrams
            if all(word.pos_ in include_pos_ for word in ngram)
        )
    for ngram in ngrams:
        yield ngram


def get_pattern_matching_candidates(
    doc: Doc,
    patterns: str | list[str] | list[dict] | list[list[dict]],
) -> Iterable[tuple[Token, ...]]:
    """
    Get candidate keyterms from ``doc``, where candidates are sequences of tokens
    that match any pattern in ``patterns``

    Args:
        doc
        patterns: One or multiple patterns to match against ``doc`` using
            a :class:`spacy.matcher.Matcher`.

    Yields:
        tuple[:class:`spacy.tokens.Token`]: Next pattern-matching candidate,
        as a tuple of constituent Tokens.

    See Also:
        :func:`textacy.extract.token_matches()`
    """
    for match in matches.token_matches(doc, patterns, on_match=None):
        yield tuple(match)


def get_filtered_topn_terms(
    term_scores: Iterable[tuple[str, float]],
    topn: int,
    *,
    match_threshold: Optional[float] = None,
) -> list[tuple[str, float]]:
    """
    Build up a list of the ``topn`` terms, filtering out any that are substrings
    of better-scoring terms and optionally filtering out any that are sufficiently
    similar to better-scoring terms.

    Args:
        term_scores: Iterable of (term, score) pairs, sorted in order of score
            from best to worst. Note that this may be from high to low value or
            low to high, depending on the scoring algorithm.
        topn: Maximum number of top-scoring terms to get.
        match_threshold: Minimal edit distance between a term and previously seen terms,
            used to filter out terms that are sufficiently similar
            to higher-scoring terms. Uses :func:`textacy.similarity.token_sort_ratio()`.
    """
    from .. import similarity  # ugh, hide import here

    topn_terms = []
    seen_terms: set[str] = set()
    sim_func = similarity.token_sort_ratio
    for term, score in term_scores:
        # skip terms that are substrings of any higher-scoring term
        if any(term in st for st in seen_terms):
            continue
        # skip terms that are sufficiently similar to any higher-scoring term
        if match_threshold and any(
            sim_func(term, st) >= match_threshold for st in seen_terms
        ):
            continue
        seen_terms.add(term)
        topn_terms.append((term, score))
        if len(topn_terms) >= topn:
            break
    return topn_terms


# TODO: decide if we want to keep this function, and if so, where to put it

# import math
# from operator import itemgetter
# from decimal import Decimal

# import numpy as np

# from .. import vsm


# def most_discriminating_terms(
#     terms_lists: Iterable[Iterable[str]],
#     bool_array_grp1: Iterable[bool],
#     *,
#     max_n_terms: int = 1000,
#     top_n_terms: int | float = 25,
# ) -> tuple[list[str], list[str]]:
#     """
#     Given a collection of documents assigned to 1 of 2 exclusive groups, get the
#     ``top_n_terms`` most discriminating terms for group1-and-not-group2 and
#     group2-and-not-group1.

#     Args:
#         terms_lists: Sequence of documents, each as a sequence of (str) terms;
#             used as input to :func:`doc_term_matrix()`
#         bool_array_grp1: Ordered sequence of True/False values,
#             where True corresponds to documents falling into "group 1" and False
#             corresponds to those in "group 2".
#         max_n_terms: Only consider terms whose document frequency is within
#             the top ``max_n_terms`` out of all distinct terms; must be > 0.
#         top_n_terms: If int (must be > 0), the total number of most discriminating terms
#             to return for each group; if float (must be in the interval (0, 1)),
#             the fraction of ``max_n_terms`` to return for each group.

#     Returns:
#         List of the top ``top_n_terms`` most discriminating terms for grp1-not-grp2, and
#         list of the top ``top_n_terms`` most discriminating terms for grp2-not-grp1.

#     References:
#         King, Gary, Patrick Lam, and Margaret Roberts. "Computer-Assisted Keyword
#         and Document Set Discovery from Unstructured Text." (2014).
#         http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.1445&rep=rep1&type=pdf
#     """
#     alpha_grp1 = 1
#     alpha_grp2 = 1
#     if isinstance(top_n_terms, float):
#         top_n_terms = int(round(top_n_terms * max_n_terms))
#     bool_array_grp1 = np.array(bool_array_grp1)
#     bool_array_grp2 = np.invert(bool_array_grp1)

#     vectorizer = vsm.Vectorizer(
#         tf_type="linear",
#         norm=None,
#         idf_type="smooth",
#         min_df=3,
#         max_df=0.95,
#         max_n_terms=max_n_terms,
#     )
#     dtm = vectorizer.fit_transform(terms_lists)
#     id2term = vectorizer.id_to_term

#     # get doc freqs for all terms in grp1 documents
#     dtm_grp1 = dtm[bool_array_grp1, :]
#     n_docs_grp1 = dtm_grp1.shape[0]
#     doc_freqs_grp1 = vsm.get_doc_freqs(dtm_grp1)

#     # get doc freqs for all terms in grp2 documents
#     dtm_grp2 = dtm[bool_array_grp2, :]
#     n_docs_grp2 = dtm_grp2.shape[0]
#     doc_freqs_grp2 = vsm.get_doc_freqs(dtm_grp2)

#     # get terms that occur in a larger fraction of grp1 docs than grp2 docs
#     term_ids_grp1 = np.where(
#         doc_freqs_grp1 / n_docs_grp1 > doc_freqs_grp2 / n_docs_grp2
#     )[0]

#     # get terms that occur in a larger fraction of grp2 docs than grp1 docs
#     term_ids_grp2 = np.where(
#         doc_freqs_grp1 / n_docs_grp1 < doc_freqs_grp2 / n_docs_grp2
#     )[0]

#     # get grp1 terms doc freqs in and not-in grp1 and grp2 docs, plus marginal totals
#     grp1_terms_grp1_df = doc_freqs_grp1[term_ids_grp1]
#     grp1_terms_grp2_df = doc_freqs_grp2[term_ids_grp1]
#     # grp1_terms_grp1_not_df = n_docs_grp1 - grp1_terms_grp1_df
#     # grp1_terms_grp2_not_df = n_docs_grp2 - grp1_terms_grp2_df
#     # grp1_terms_total_df = grp1_terms_grp1_df + grp1_terms_grp2_df
#     # grp1_terms_total_not_df = grp1_terms_grp1_not_df + grp1_terms_grp2_not_df

#     # get grp2 terms doc freqs in and not-in grp2 and grp1 docs, plus marginal totals
#     grp2_terms_grp2_df = doc_freqs_grp2[term_ids_grp2]
#     grp2_terms_grp1_df = doc_freqs_grp1[term_ids_grp2]
#     # grp2_terms_grp2_not_df = n_docs_grp2 - grp2_terms_grp2_df
#     # grp2_terms_grp1_not_df = n_docs_grp1 - grp2_terms_grp1_df
#     # grp2_terms_total_df = grp2_terms_grp2_df + grp2_terms_grp1_df
#     # grp2_terms_total_not_df = grp2_terms_grp2_not_df + grp2_terms_grp1_not_df

#     # get grp1 terms likelihoods, then sort for most discriminating grp1-not-grp2 terms
#     grp1_terms_likelihoods = {}
#     for idx, term_id in enumerate(term_ids_grp1):
#         term1 = (
#             Decimal(math.factorial(grp1_terms_grp1_df[idx] + alpha_grp1 - 1))
#             * Decimal(math.factorial(grp1_terms_grp2_df[idx] + alpha_grp2 - 1))
#             / Decimal(
#                 math.factorial(
#                     grp1_terms_grp1_df[idx]
#                     + grp1_terms_grp2_df[idx]
#                     + alpha_grp1
#                     + alpha_grp2
#                     - 1
#                 )
#             )
#         )
#         term2 = (
#             Decimal(
#                 math.factorial(n_docs_grp1 - grp1_terms_grp1_df[idx] + alpha_grp1 - 1)
#             )
#             * Decimal(
#                 math.factorial(n_docs_grp2 - grp1_terms_grp2_df[idx] + alpha_grp2 - 1)
#             )
#             / Decimal(
#                 (
#                     math.factorial(
#                         n_docs_grp1
#                         + n_docs_grp2
#                         - grp1_terms_grp1_df[idx]
#                         - grp1_terms_grp2_df[idx]
#                         + alpha_grp1
#                         + alpha_grp2
#                         - 1
#                     )
#                 )
#             )
#         )
#         grp1_terms_likelihoods[id2term[term_id]] = term1 * term2
#     top_grp1_terms = [
#         term
#         for term, likelihood in sorted(
#             grp1_terms_likelihoods.items(), key=itemgetter(1), reverse=True
#         )[:top_n_terms]
#     ]

#     # get grp2 terms likelihoods, then sort for most discriminating grp2-not-grp1 terms
#     grp2_terms_likelihoods = {}
#     for idx, term_id in enumerate(term_ids_grp2):
#         term1 = (
#             Decimal(math.factorial(grp2_terms_grp2_df[idx] + alpha_grp2 - 1))
#             * Decimal(math.factorial(grp2_terms_grp1_df[idx] + alpha_grp1 - 1))
#             / Decimal(
#                 math.factorial(
#                     grp2_terms_grp2_df[idx]
#                     + grp2_terms_grp1_df[idx]
#                     + alpha_grp2
#                     + alpha_grp1
#                     - 1
#                 )
#             )
#         )
#         term2 = (
#             Decimal(
#                 math.factorial(n_docs_grp2 - grp2_terms_grp2_df[idx] + alpha_grp2 - 1)
#             )
#             * Decimal(
#                 math.factorial(n_docs_grp1 - grp2_terms_grp1_df[idx] + alpha_grp1 - 1)
#             )
#             / Decimal(
#                 (
#                     math.factorial(
#                         n_docs_grp2
#                         + n_docs_grp1
#                         - grp2_terms_grp2_df[idx]
#                         - grp2_terms_grp1_df[idx]
#                         + alpha_grp2
#                         + alpha_grp1
#                         - 1
#                     )
#                 )
#             )
#         )
#         grp2_terms_likelihoods[id2term[term_id]] = term1 * term2
#     top_grp2_terms = [
#         term
#         for term, likelihood in sorted(
#             grp2_terms_likelihoods.items(), key=itemgetter(1), reverse=True
#         )[:top_n_terms]
#     ]

#     return (top_grp1_terms, top_grp2_terms)
