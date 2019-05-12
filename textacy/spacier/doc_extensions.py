# -*- coding: utf-8 -*-
"""
spaCy Doc extensions
--------------------

Functionality for inspecting, customizing, and transforming spaCy's core
data structure, :class:`spacy.tokens.Doc`, accessible directly as functions
that take a ``Doc`` as their first argument or as custom attributes/methods
on instantiated docs prepended by an underscore:

.. code-block:: pycon

    >>> spacy_lang = textacy.load_spacy_lang("en")
    >>> doc = nlp("This is a short text.")
    >>> print(get_preview(doc))
    Doc(6 tokens: "This is a short text.")
    >>> print(doc._.preview)
    Doc(6 tokens: "This is a short text.")
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import spacy
from cytoolz import itertoolz

from .. import constants
from .. import extract
from .. import network

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


def get_lang(doc):
    """
    Get the standard, two-letter language code assigned to ``doc``
    and its associated :class:`spacy.vocab.Vocab`.

    Returns:
        str
    """
    return doc.vocab.lang


def get_preview(doc):
    """
    Get a short preview of the ``doc``, including the number of tokens
    and an initial snippet.

    Returns:
        str
    """
    snippet = doc.text[:50].replace("\n", " ")
    if len(snippet) == 50:
        snippet = snippet[:47] + "..."
    return 'Doc({} tokens: "{}")'.format(len(doc), snippet)


def get_tokens(doc):
    """
    Yield the tokens in ``doc``, one at a time.

    Yields:
        :class:`spacy.tokens.Token`
    """
    for tok in doc:
        yield tok


def get_n_tokens(doc):
    """
    Get the number of tokens (including punctuation) in ``doc``.

    Returns:
        int
    """
    return len(doc)


def get_n_sents(doc):
    """
    Get the number of sentences in ``doc``.

    Returns:
        int
    """
    return sum(1 for _ in doc.sents)


def get_meta(doc):
    """
    Get custom metadata added to ``doc``.

    Returns:
        dict
    """
    return doc.user_data.get("textacy", {}).get("meta", {})


def set_meta(doc, value):
    """
    Add custom metadata to ``doc``.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        value (dict)
    """
    if not isinstance(value, dict):
        raise TypeError("doc metadata must be a dict, not {}".format(type(value)))
    try:
        doc.user_data["textacy"]["meta"] = value
    except KeyError:
        doc.user_data["textacy"] = {}
        doc.user_data["textacy"]["meta"] = value


def to_tokenized_text(doc):
    """
    Transform ``doc`` into an ordered, nested list of token-texts per sentence.

    Returns:
        List[List[str]]

    Note:
        If ``doc`` hasn't been segmented into sentences, the entire document
        is treated as a single sentence.
    """
    if doc.is_sentenced:
        return [
            [token.text for token in sent]
            for sent in doc.sents
        ]
    else:
        return [[token.text for token in doc]]


def to_tagged_text(doc):
    """
    Transform ``doc`` into an ordered, nested list of (token-text, part-of-speech tag)
    pairs per sentence.

    Returns:
        List[List[Tuple[str, str]]]

    Note:
        If ``doc`` hasn't been segmented into sentences, the entire document
        is treated as a single sentence.
    """
    if doc.is_sentenced:
        return [
            [(token.text, token.pos_) for token in sent]
            for sent in doc.sents
        ]
    else:
        return [[(token.text, token.pos_) for token in doc]]


def to_terms_list(
    doc,
    ngrams=(1, 2, 3),
    entities=True,
    normalize="lemma",
    as_strings=False,
    **kwargs
):
    """
    Transform :class:`Doc` into a sequence of ngrams and/or named entities, which
    aren't necessarily in order of appearance, where each term appears in
    the list with the same frequency that it appears in :class:`Doc`.

    Args:
        ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
            (default) includes unigrams (words), bigrams, and trigrams; `2`
            if only bigrams are wanted; falsy (e.g. False) to not include any
        entities (bool): if True (default), include named entities
            in the terms list; note: if ngrams are also included, named
            entities are added *first*, and any ngrams that exactly overlap
            with an entity are skipped to prevent double-counting
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if false-y, use the form of terms as they appear
            in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
            and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        as_strings (bool): if True, terms are returned as strings; if False
            (default), terms are returned as their unique integer ids
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

            see :func:`extract.words <textacy.extract.words>`,
            :func:`extract.ngrams <textacy.extract.ngrams>`,
            and :func:`extract.entities <textacy.extract.entities>`
            for more information on these parameters

    Yields:
        int or str: the next term in the terms list, either as a unique
        integer id or as a string

    Raises:
        ValueError: if neither ``entities`` nor ``ngrams`` are included

    Note:
        Despite the name, this is a generator function; to get an
        actual list of terms, call ``list(doc.to_terms_list())``.
    """
    if not (entities or ngrams):
        raise ValueError("either `entities` or `ngrams` must be included")
    if ngrams and isinstance(ngrams, int):
        ngrams = (ngrams,)
    if entities is True:
        ne_kwargs = {
            "include_types": kwargs.get("include_types"),
            "exclude_types": kwargs.get("exclude_types"),
            "drop_determiners": kwargs.get("drop_determiners", True),
            "min_freq": kwargs.get("min_freq", 1),
        }
        # if numeric ngrams are to be filtered, we should filter numeric entities
        if ngrams and kwargs.get("filter_nums") is True:
            if ne_kwargs["exclude_types"]:
                if isinstance(
                    ne_kwargs["exclude_types"], (set, frozenset, list, tuple)
                ):
                    ne_kwargs["exclude_types"] = set(ne_kwargs["exclude_types"])
                    ne_kwargs["exclude_types"].add(constants.NUMERIC_ENT_TYPES)
            else:
                ne_kwargs["exclude_types"] = constants.NUMERIC_ENT_TYPES
    if ngrams:
        ngram_kwargs = {
            "filter_stops": kwargs.get("filter_stops", True),
            "filter_punct": kwargs.get("filter_punct", True),
            "filter_nums": kwargs.get("filter_nums", False),
            "include_pos": kwargs.get("include_pos"),
            "exclude_pos": kwargs.get("exclude_pos"),
            "min_freq": kwargs.get("min_freq", 1),
        }
        # if numeric entities are to be filtered, we should filter numeric ngrams
        if (
            entities
            and ne_kwargs["exclude_types"]
            and any(
                ent_type in ne_kwargs["exclude_types"]
                for ent_type in constants.NUMERIC_ENT_TYPES
            )
        ):
            ngram_kwargs["filter_nums"] = True

    terms = []
    # special case: ensure that named entities aren't double-counted when
    # adding words or ngrams that were already added as named entities
    if entities is True and ngrams:
        ents = tuple(extract.entities(doc, **ne_kwargs))
        ent_idxs = {(ent.start, ent.end) for ent in ents}
        terms.append(ents)
        for n in ngrams:
            if n == 1:
                terms.append(
                    (
                        word
                        for word in extract.words(doc, **ngram_kwargs)
                        if (word.i, word.i + 1) not in ent_idxs
                    )
                )
            else:
                terms.append(
                    (
                        ngram
                        for ngram in extract.ngrams(doc, n, **ngram_kwargs)
                        if (ngram.start, ngram.end) not in ent_idxs
                    )
                )
    # otherwise, no need to check for overlaps
    else:
        if entities is True:
            terms.append(extract.entities(doc, **ne_kwargs))
        else:
            for n in ngrams:
                if n == 1:
                    terms.append(extract.words(doc, **ngram_kwargs))
                else:
                    terms.append(extract.ngrams(doc, n, **ngram_kwargs))

    terms = itertoolz.concat(terms)

    # convert token and span objects into integer ids
    if as_strings is False:
        if normalize == "lemma":
            for term in terms:
                try:
                    yield term.lemma
                except AttributeError:
                    yield doc.vocab.strings.add(term.lemma_)
        elif normalize == "lower":
            for term in terms:
                try:
                    yield term.lower
                except AttributeError:
                    yield doc.vocab.strings.add(term.lower_)
        elif not normalize:
            for term in terms:
                try:
                    yield term.orth
                except AttributeError:
                    yield doc.vocab.strings.add(term.text)
        else:
            for term in terms:
                yield doc.vocab.strings.add(normalize(term))

    # convert token and span objects into strings
    else:
        if normalize == "lemma":
            for term in terms:
                yield term.lemma_
        elif normalize == "lower":
            for term in terms:
                yield term.lower_
        elif not normalize:
            for term in terms:
                yield term.text
        else:
            for term in terms:
                yield normalize(term)


def to_bag_of_terms(
    doc,
    ngrams=(1, 2, 3),
    entities=True,
    normalize="lemma",
    weighting="count",
    as_strings=False,
    **kwargs
):
    """
    Transform :class:`Doc` into a bag-of-terms: the set of unique terms in
    :class:`Doc` mapped to their frequency of occurrence, where "terms"
    includes ngrams and/or named entities.

    Args:
        ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
            (default) includes unigrams (words), bigrams, and trigrams; `2`
            if only bigrams are wanted; falsy (e.g. False) to not include any
        entities (bool): if True (default), include named entities;
            note: if ngrams are also included, any ngrams that exactly
            overlap with an entity are skipped to prevent double-counting
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if false-y, use the form of terms as they appear
            in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
            and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
            terms. If 'count' (default), weights are the absolute number of
            occurrences (count) of term in doc. If 'binary', all counts are
            set equal to 1. If 'freq', term counts are normalized by the
            total token count, giving their relative frequency of occurrence.
        as_strings (bool): if True, words are returned as strings; if False
            (default), words are returned as their unique integer ids
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

            See :func:`extract.words() <textacy.extract.words>`,
            :func:`extract.ngrams() <textacy.extract.ngrams>`,
            and :func:`extract.entities() <textacy.extract.entities>`
            for more information on these parameters.

    Returns:
        dict: mapping of a unique term id or string (depending on the value
        of ``as_strings``) to its absolute, relative, or binary frequency
        of occurrence (depending on the value of ``weighting``).

    See Also:
        :meth:`Doc.to_terms_list() <Doc.to_terms_list>`
    """
    if weighting not in {"count", "freq", "binary"}:
        raise ValueError('weighting "{}" is invalid'.format(weighting))
    terms_list = to_terms_list(
        doc,
        ngrams=ngrams,
        entities=entities,
        normalize=normalize,
        as_strings=as_strings,
        **kwargs
    )
    bot = itertoolz.frequencies(terms_list)
    if weighting == "freq":
        n_tokens = len(doc)
        bot = {term: weight / n_tokens for term, weight in bot.items()}
    elif weighting == "binary":
        bot = {term: 1 for term in bot.keys()}
    return bot


def to_bag_of_words(doc, normalize="lemma", weighting="count", as_strings=False):
    """
    Transform :class:`Doc` into a bag-of-words: the set of unique words in
    :class:`Doc` mapped to their absolute, relative, or binary frequency of
    occurrence.

    Args:
        normalize (str): if 'lemma', lemmatize words before counting; if
            'lower', lowercase words before counting; otherwise, words are
            counted using the form with which they they appear in doc
        weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
            words. If 'count' (default), weights are the absolute number of
            occurrences (count) of word in doc. If 'binary', all counts are
            set equal to 1. If 'freq', word counts are normalized by the
            total token count, giving their relative frequency of occurrence.
            Note: The resulting set of frequencies won't (necessarily) sum
            to 1.0, since punctuation and stop words are filtered out after
            counts are normalized.
        as_strings (bool): if True, words are returned as strings; if False
            (default), words are returned as their unique integer ids

    Returns:
        dict: mapping of a unique word id or string (depending on the value
        of ``as_strings``) to its absolute, relative, or binary frequency
        of occurrence (depending on the value of ``weighting``).
    """
    if weighting not in {"count", "freq", "binary"}:
        raise ValueError('weighting "{}" is invalid'.format(weighting))
    count_by = (
        spacy.attrs.LEMMA if normalize == "lemma"
        else spacy.attrs.LOWER if normalize == "lower"
        else spacy.attrs.ORTH
    )
    word_to_weight = doc.count_by(count_by)
    if weighting == "freq":
        n_tokens = len(doc)
        word_to_weight = {
            id_: weight / n_tokens for id_, weight in word_to_weight.items()
        }
    elif weighting == "binary":
        word_to_weight = {word: 1 for word in word_to_weight.keys()}

    bow = {}
    if as_strings is False:
        for id_, weight in word_to_weight.items():
            lexeme = doc.vocab[id_]
            if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                continue
            bow[id_] = weight
    else:
        for id_, weight in word_to_weight.items():
            lexeme = doc.vocab[id_]
            if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                continue
            bow[doc.vocab.strings[id_]] = weight
    return bow


def to_semantic_network(
    doc,
    nodes="words",
    normalize="lemma",
    edge_weighting="default",
    window_width=10,
):
    """
    Transform :class:`Doc` into a semantic network, where nodes are either
    'words' or 'sents' and edges between nodes may be weighted in different ways.

    Args:
        nodes ({'words', 'sents'}): type of doc component to use as nodes
            in the semantic network
        normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
            lowercase terms; if false-y, use the form of terms as they appear
            in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
            (if ``nodes`` = 'words' or 'sents', respectively) and return a
            str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
        edge_weighting (str): type of weighting to apply to edges
            between nodes; if ``nodes == 'words'``, options are {'cooc_freq', 'binary'},
            if ``nodes == 'sents'``, options are {'cosine', 'jaccard'}; if
            'default', 'cooc_freq' or 'cosine' will be automatically used
        window_width (int): size of sliding window over terms that
            determines which are said to co-occur; only applicable if 'words'

    Returns:
        ``networkx.Graph``: where nodes represent either terms or sentences
        in doc; edges, the relationships between them.

    Raises:
        ValueError: if ``nodes`` is neither 'words' nor 'sents'.

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
        msg = 'nodes "{}" not valid; must be in {}'.format(
            nodes, {"words", "sents"}
        )
        raise ValueError(msg)


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
