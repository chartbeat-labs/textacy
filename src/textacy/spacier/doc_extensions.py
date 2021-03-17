"""
:mod:`textacy.spacier.doc_extensions`: Inspect, extend, and transform spaCy's core
data structure, :class:`spacy.tokens.Doc`, either directly via functions that take
a ``Doc`` as their first argument or as custom attributes / methods on instantiated docs
prepended by an underscore:

.. code-block:: pycon

    >>> spacy_lang = textacy.load_spacy_lang("en")
    >>> doc = spacy_lang("This is a short text.")
    >>> print(get_preview(doc))
    Doc(6 tokens: "This is a short text.")
    >>> print(doc._.preview)
    Doc(6 tokens: "This is a short text.")
"""
from typing import (
    cast,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx
import spacy
from cytoolz import itertoolz
from spacy.tokens import Doc, Span, Token

from .. import errors
from .. import extract
from .. import network
from .. import utils

__all__ = [
    "set_doc_extensions",
    "get_doc_extensions",
    "remove_doc_extensions",
    "get_lang",
    "get_preview",
    "get_tokens",
    "get_meta",
    "set_meta",
    "get_n_tokens",
    "get_n_sents",
    "to_tokenized_text",
    "to_tagged_text",
    "to_terms_list",
    "to_bag_of_terms",
    "to_bag_of_words",
    "to_semantic_network",
]


def set_doc_extensions():
    """
    Set textacy's custom property and method doc extensions
    on the global :class:`spacy.tokens.Doc`.
    """
    for name, kwargs in _doc_extensions.items():
        if not spacy.tokens.Doc.has_extension(name):
            spacy.tokens.Doc.set_extension(name, **kwargs)


def get_doc_extensions():
    """
    Get textacy's custom property and method doc extensions
    that can be set on or removed from the global :class:`spacy.tokens.Doc`.
    """
    return _doc_extensions


def remove_doc_extensions():
    """
    Remove textacy's custom property and method doc extensions
    from the global :class:`spacy.tokens.Doc`.
    """
    for name in _doc_extensions.keys():
        _ = spacy.tokens.Doc.remove_extension(name)


def get_lang(doc: Doc) -> str:
    """
    Get the standard, two-letter language code assigned to ``Doc``
    and its associated :class:`spacy.vocab.Vocab`.
    """
    return doc.vocab.lang


def get_preview(doc: Doc) -> str:
    """
    Get a short preview of the ``Doc``, including the number of tokens
    and an initial snippet.
    """
    snippet = doc.text[:50].replace("\n", " ")
    if len(snippet) == 50:
        snippet = snippet[:47] + "..."
    return 'Doc({} tokens: "{}")'.format(len(doc), snippet)


def get_tokens(doc: Doc) -> Iterable[Token]:
    """Yield the tokens in ``Doc``, one at a time."""
    for tok in doc:
        yield tok


def get_n_tokens(doc: Doc) -> int:
    """Get the number of tokens (including punctuation) in ``Doc``."""
    return len(doc)


def get_n_sents(doc: Doc) -> int:
    """Get the number of sentences in ``Doc``."""
    return itertoolz.count(doc.sents)


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
        doc.user_data["textacy"] = {}
        doc.user_data["textacy"]["meta"] = value


def to_tokenized_text(doc: Doc) -> List[List[str]]:
    """
    Transform ``Doc`` into an ordered, nested list of token-texts per sentence.

    Note:
        If ``doc`` hasn't been segmented into sentences, the entire document
        is treated as a single sentence.
    """
    if doc.has_annotation("SENT_START"):
        return [[token.text for token in sent] for sent in doc.sents]
    else:
        return [[token.text for token in doc]]


def to_tagged_text(doc: Doc) -> List[List[Tuple[str, str]]]:
    """
    Transform ``Doc`` into an ordered, nested list of (token-text, part-of-speech tag)
    pairs per sentence.

    Note:
        If ``doc`` hasn't been segmented into sentences, the entire document
        is treated as a single sentence.
    """
    if doc.has_annotation("SENT_START"):
        return [[(token.text, token.pos_) for token in sent] for sent in doc.sents]
    else:
        return [[(token.text, token.pos_) for token in doc]]


def to_terms_list(
    doc: Doc,
    *,
    ngrams: Optional[Union[int, Collection[int]]] = (1, 2, 3),
    entities: Optional[bool] = True,
    normalize: Optional[Union[str, Callable[[Union[Span, Token]], str]]] = "lemma",
    as_strings: bool = False,
    **kwargs,
) -> Union[Iterable[int], Iterable[str]]:
    """
    Transform ``Doc`` into a sequence of ngrams and/or entities — not necessarily
    in order of appearance — where each appears in the sequence as many times as
    it appears in ``Doc``.

    Args:
        doc
        ngrams: ngrams to include in the terms list.
            If ``{1, 2, 3}``, unigrams, bigrams, and trigrams are included;
            if ``2``, only bigrams are included; if None, ngrams aren't included,
            except for those belonging to named entities.
        entities: If True, entities are included in the terms list;
            if False, they are *excluded* from the list;
            if None, entities aren't included or excluded at all.

            .. note:: When both ``entities`` and ``ngrams`` are non-null,
               exact duplicates (based on start and end indexes) are handled.
               If ``entities`` is True, any duplicate entities are included
               while duplicate ngrams are discarded to avoid double-counting;
               if ``entities`` is False, no entities are included of course,
               and duplicate ngrams are discarded as well.

        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms;
            if falsy, use the form of terms as they appear in doc;
            if callable, must accept a ``Token`` or ``Span`` and return a str,
            e.g. :func:`get_normalized_text() <textacy.spacier.utils.get_normalized_text>`.
        as_strings: If True, terms are returned as strings;
            if False, terms are returned as their unique integer ids.
        kwargs:
            - filter_stops (bool)
            - filter_punct (bool)
            - filter_nums (bool)
            - include_pos (str or Set[str])
            - exclude_pos (str or Set[str])
            - min_freq (int)
            - include_types (str or Set[str])
            - exclude_types (str or Set[str]
            - drop_determiners (bool)

            See :func:`textacy.extract.words()`, :func:`textacy.extract.ngrams()`,
            and :func:`textacy.extract.entities()` for details.

    Yields:
        The next term in the terms list, as either a unique integer id or a string.

    Raises:
        ValueError: if neither ``entities`` nor ``ngrams`` are included,
            or if ``normalize`` have invalid values
        TypeError: if ``entities`` has an invalid type

    Note:
        Despite the name, this is a generator function; to get an
        actual list of terms, call ``list(to_terms_list(doc))``.
    """
    if not (entities or ngrams):
        raise ValueError("`entities` and/or `ngrams` must be included")
    if not (entities is None or isinstance(entities, bool)):
        raise TypeError(
            errors.type_invalid_msg("entities", type(entities), Optional[bool])
        )
    if not (normalize in ("lemma", "lower") or callable(normalize) or not normalize):
        raise ValueError(
            errors.value_invalid_msg(
                "normalize", normalize, {"lemma", "lower", None, Callable}
            )
        )
    if ngrams:
        ngrams = cast(Set[int], utils.to_collection(ngrams, int, set))
        unigrams_: Iterable[Token] = []
        ngrams_: List[Iterable[Span]] = []
        ng_kwargs = utils.get_kwargs_for_func(extract.ngrams, kwargs)
        for n in sorted(ngrams):
            # use a faster function for unigrams
            if n == 1:
                unigrams_ = extract.words(doc, **ng_kwargs)
            else:
                ngrams_.append(extract.ngrams(doc, n, **ng_kwargs))
        ngrams_ = itertoolz.concat(ngrams_)
    if entities is not None:
        ent_kwargs = utils.get_kwargs_for_func(extract.entities, kwargs)
        entities_ = extract.entities(doc, **ent_kwargs)
    if ngrams:
        # use ngrams as-is
        if entities is None:
            terms = itertoolz.concatv(unigrams_, ngrams_)
        # remove unigrams + ngrams that are duplicates of entities
        else:
            entities_ = tuple(entities_)
            ent_idxs = {(ent.start, ent.end) for ent in entities_}
            unigrams_ = (ug for ug in unigrams_ if (ug.i, ug.i + 1) not in ent_idxs)
            ngrams_ = (ng for ng in ngrams_ if (ng.start, ng.end) not in ent_idxs)
            # add unigrams and ngrams, only
            if entities is False:
                terms = itertoolz.concatv(unigrams_, ngrams_)
            # add unigrams, ngrams, and entities
            else:
                terms = itertoolz.concatv(unigrams_, ngrams_, entities_)
    # use entities as-is
    else:
        terms = entities_

    # convert spans into integer ids
    if as_strings is False:
        ss = doc.vocab.strings
        if normalize == "lemma":
            for term in terms:
                try:
                    yield term.lemma
                except AttributeError:
                    yield ss.add(term.lemma_)
        elif normalize == "lower":
            for term in terms:
                try:
                    yield term.lower
                except AttributeError:
                    yield ss.add(term.lower_)
        elif callable(normalize):
            for term in terms:
                yield ss.add(normalize(term))
        else:
            for term in terms:
                try:
                    yield term.text
                except AttributeError:
                    yield ss.add(term.text)
    # convert spans into strings
    else:
        if normalize == "lemma":
            for term in terms:
                yield term.lemma_
        elif normalize == "lower":
            for term in terms:
                yield term.lower_
        elif callable(normalize):
            for term in terms:
                yield normalize(term)
        else:
            for term in terms:
                yield term.text


def to_bag_of_terms(
    doc: Doc,
    *,
    ngrams: Optional[Union[int, Collection[int]]] = (1, 2, 3),
    entities: Optional[bool] = True,
    normalize: Optional[Union[str, Callable[[Union[Span, Token]], str]]] = "lemma",
    weighting: str = "count",
    as_strings: bool = False,
    **kwargs,
) -> Dict[Union[int, str], Union[int, float]]:
    """
    Transform ``Doc`` into a bag-of-terms: the set of unique terms in ``Doc``
    mapped to their frequency of occurrence, where "terms" includes ngrams and/or entities.

    Args:
        doc
        ngrams: n of which n-grams to include.
            ``(1, 2, 3)`` (default) includes unigrams (words), bigrams, and trigrams;
            `2` if only bigrams are wanted; falsy (e.g. False) to not include any
        entities: If True (default), include named entities;
            note: if ngrams are also included, any ngrams that exactly
            overlap with an entity are skipped to prevent double-counting
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms; if falsy,
            use the form of terms as they appear in ``doc``;
            if a callable, must accept a ``Token`` or ``Span`` and return a str,
            e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        weighting ({"count", "freq", "binary"}): Type of weight to assign to
            terms. If "count" (default), weights are the absolute number of
            occurrences (count) of term in doc. If "binary", all counts are
            set equal to 1. If "freq", term counts are normalized by the
            total token count, giving their relative frequency of occurrence.
        as_strings: If True, words are returned as strings;
            if False (default), words are returned as their unique integer ids.
        kwargs:
            - filter_stops (bool)
            - filter_punct (bool)
            - filter_nums (bool)
            - include_pos (str or Set[str])
            - exclude_pos (str or Set[str])
            - min_freq (int)
            - include_types (str or Set[str])
            - exclude_types (str or Set[str]
            - drop_determiners (bool)

            See :func:`textacy.extract.words()`, :func:`textacy.extract.ngrams()`,
            and :func:`textacy.extract.entities()`  for details.

    Returns:
        Mapping of a unique term id or string (depending on the value of ``as_strings``)
        to its absolute, relative, or binary frequency of occurrence
        (depending on the value of ``weighting``).

    See Also:
        :func:`to_terms_list()`, which is used under the hood.
    """
    if weighting not in {"count", "freq", "binary"}:
        raise ValueError(
            errors.value_invalid_msg("weighting", weighting, {"count", "freq", "binary"})
        )
    terms_list = to_terms_list(
        doc,
        ngrams=ngrams,
        entities=entities,
        normalize=normalize,
        as_strings=as_strings,
        **kwargs,
    )
    bot = itertoolz.frequencies(terms_list)
    if weighting == "freq":
        n_tokens = len(doc)
        bot = {term: weight / n_tokens for term, weight in bot.items()}
    elif weighting == "binary":
        bot = {term: 1 for term in bot.keys()}
    return bot


def to_bag_of_words(
    doc: Doc,
    *,
    normalize: str = "lemma",
    weighting: str = "count",
    as_strings: bool = False,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
) -> Dict[Union[int, str], Union[int, float]]:
    """
    Transform ``Doc`` into a bag-of-words: the set of unique words in ``Doc``
    mapped to their absolute, relative, or binary frequency of occurrence.

    Args:
        doc
        normalize: If "lemma", lemmatize words before counting;
            if "lower", lowercase words before counting; otherwise, words are
            counted using the form with which they they appear in doc.
        weighting ({"count", "freq", "binary"}): Type of weight to assign to
            words. If "count" (default), weights are the absolute number of
            occurrences (count) of word in doc. If "binary", all counts are
            set equal to 1. If "freq", word counts are normalized by the
            total token count, giving their relative frequency of occurrence.
            Note: The resulting set of frequencies won't (necessarily) sum
            to 1.0, since punctuation and stop words are filtered out after
            counts are normalized.
        as_strings (bool): If True, words are returned as strings; if False
            (default), words are returned as their unique integer ids
        filter_stops (bool): If True (default), stop words are removed after
            counting.
        filter_punct (bool): If True (default), punctuation tokens are removed
            after counting.
        filter_nums (bool): If True, tokens consisting of digits are removed
            after counting.

    Returns:
        Mapping of a unique term id or string (depending on the value of ``as_strings``)
        to its absolute, relative, or binary frequency of occurrence
        (depending on the value of ``weighting``).
    """
    if weighting not in {"count", "freq", "binary"}:
        raise ValueError(
            errors.value_invalid_msg("weighting", weighting, {"count", "freq", "binary"})
        )
    count_by = (
        spacy.attrs.LEMMA
        if normalize == "lemma"
        else spacy.attrs.LOWER
        if normalize == "lower"
        else spacy.attrs.ORTH
    )

    wid_weights = doc.count_by(count_by)
    if weighting == "freq":
        n_tokens = len(doc)
        wid_weights = {wid: weight / n_tokens for wid, weight in wid_weights.items()}
    elif weighting == "binary":
        wid_weights = {wid: 1 for wid in wid_weights.keys()}

    bow = {}
    vocab = doc.vocab
    if as_strings is False:
        for wid, weight in wid_weights.items():
            lex = vocab[wid]
            if not (
                (lex.is_stop and filter_stops)
                or (lex.is_punct and filter_punct)
                or (lex.is_digit and filter_nums)
                or lex.is_space
            ):
                bow[wid] = weight
    else:
        ss = doc.vocab.strings
        for wid, weight in wid_weights.items():
            lex = vocab[wid]
            if not (
                (lex.is_stop and filter_stops)
                or (lex.is_punct and filter_punct)
                or (lex.is_digit and filter_nums)
                or lex.is_space
            ):
                bow[ss[wid]] = weight
    return bow


def to_semantic_network(
    doc: Doc,
    *,
    nodes: str = "words",
    normalize: Optional[Union[str, Callable[[Union[Span, Token]], str]]] = "lemma",
    edge_weighting: str = "default",
    window_width: int = 10,
) -> nx.Graph:
    """
    Transform ``Doc`` into a semantic network, where nodes are either
    "words" or "sents" and edges between nodes may be weighted in different ways.

    Args:
        doc
        nodes ({"words", "sents"}): Type of doc component to use as nodes
            in the semantic network.
        normalize: If "lemma", lemmatize terms; if "lower", lowercase terms;
            if falsy, use the form of terms as they appear in doc; if a callable,
            must accept a ``Token`` or ``Span``
            (if ``nodes`` = "words" or "sents", respectively) and return a str,
            e.g. :func:`get_normalized_text() <textacy.spacier.utils.get_normalized_text>`
        edge_weighting: Type of weighting to apply to edges between nodes;
            if ``nodes`` = "words", options are {"cooc_freq", "binary"},
            if ``nodes`` = "sents", options are {"cosine", "jaccard"}; if
            "default", "cooc_freq" or "cosine" will be automatically used.
        window_width: Size of sliding window over terms that determines
            which are said to co-occur; only applicable if ``nodes`` = "words".

    Returns:
        ``networkx.Graph``: where nodes represent either terms or sentences
        in doc; edges, the relationships between them.

    Raises:
        ValueError: If ``nodes`` is neither "words" nor "sents".

    See Also:
        - :func:`terms_to_semantic_network() <textacy.network.terms_to_semantic_network>`
        - :func:`sents_to_semantic_network() <textacy.network.sents_to_semantic_network>`
    """
    if nodes == "words":
        if edge_weighting == "default":
            edge_weighting = "cooc_freq"
        return network.terms_to_semantic_network(
            list(extract.words(doc)),
            normalize=normalize,
            window_width=window_width,
            edge_weighting=edge_weighting,
        )
    elif nodes == "sents":
        if edge_weighting == "default":
            edge_weighting = "cosine"
        return network.sents_to_semantic_network(
            list(doc.sents), normalize=normalize, edge_weighting=edge_weighting
        )
    else:
        raise ValueError(errors.value_invalid_msg("nodes", nodes, {"words", "sents"}))


_doc_extensions = {
    # property extensions
    "lang": {"getter": get_lang},
    "preview": {"getter": get_preview},
    "tokens": {"getter": get_tokens},
    "meta": {"getter": get_meta, "setter": set_meta},
    "n_tokens": {"getter": get_n_tokens},
    "n_sents": {"getter": get_n_sents},
    # method extensions
    "to_tokenized_text": {"method": to_tokenized_text},
    "to_tagged_text": {"method": to_tagged_text},
    "to_terms_list": {"method": to_terms_list},
    "to_bag_of_terms": {"method": to_bag_of_terms},
    "to_bag_of_words": {"method": to_bag_of_words},
    "to_semantic_network": {"method": to_semantic_network},
}
