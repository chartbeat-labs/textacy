# -*- coding: utf-8 -*-
"""
Doc extensions
--------------

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

import types

import spacy
from cytoolz import itertoolz

from .. import constants
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


def get_lang(doc):
    """
    Get the standard, two-letter language code assigned to ``Doc``
    and its associated :class:`spacy.vocab.Vocab`.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        str
    """
    return doc.vocab.lang


def get_preview(doc):
    """
    Get a short preview of the ``Doc``, including the number of tokens
    and an initial snippet.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        str
    """
    snippet = doc.text[:50].replace("\n", " ")
    if len(snippet) == 50:
        snippet = snippet[:47] + "..."
    return 'Doc({} tokens: "{}")'.format(len(doc), snippet)


def get_tokens(doc):
    """
    Yield the tokens in ``Doc``, one at a time.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Yields:
        :class:`spacy.tokens.Token`
    """
    for tok in doc:
        yield tok


def get_n_tokens(doc):
    """
    Get the number of tokens (including punctuation) in ``Doc``.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        int
    """
    return len(doc)


def get_n_sents(doc):
    """
    Get the number of sentences in ``Doc``.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        int
    """
    return itertoolz.count(doc.sents)


def get_meta(doc):
    """
    Get custom metadata added to ``Doc``.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        dict
    """
    return doc.user_data.get("textacy", {}).get("meta", {})


def set_meta(doc, value):
    """
    Add custom metadata to ``Doc``.

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
    Transform ``Doc`` into an ordered, nested list of token-texts per sentence.

    Args:
        doc (:class:`spacy.tokens.Doc`)

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
    Transform ``Doc`` into an ordered, nested list of (token-text, part-of-speech tag)
    pairs per sentence.

    Args:
        doc (:class:`spacy.tokens.Doc`)

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
    Transform ``Doc`` into a sequence of ngrams and/or entities — not necessarily
    in order of appearance — where each appears in the sequence as many times as
    it appears in ``Doc``.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (int or Set[int] or None): ngrams to include in the terms list.
            If ``{1, 2, 3}``, unigrams, bigrams, and trigrams are included;
            if ``2``, only bigrams are included; if None, ngrams aren't included,
            except for those belonging to named entities.
        entities (bool or None): If True, entities are included in the terms list;
            if False, they are *excluded* from the list; if None, entities
            aren't included or excluded at all.

            .. note:: When both ``entities`` and ``ngrams`` are non-null,
               exact duplicates (based on start and end indexes) are handled.
               If ``entities`` is True, any duplicate entities are included
               while duplicate ngrams are discarded to avoid double-counting;
               if ``entities`` is False, no entities are included of course,
               and duplicate ngrams are discarded as well.

        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if falsy, use the form of terms as they appear in doc;
            if callable, must accept a ``Token`` or ``Span`` and return a str,
            e.g. :func:`get_normalized_text() <textacy.spacier.utils.get_normalized_text>`.
        as_strings (bool): If True, terms are returned as strings;
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
        int or str: the next term in the terms list, either as a unique
        integer id or as a string

    Raises:
        ValueError: if neither ``entities`` nor ``ngrams`` are included,
            or if ``entities`` or ``normalize`` have invalid values

    Note:
        Despite the name, this is a generator function; to get an
        actual list of terms, call ``list(to_terms_list(doc))``.
    """
    if not (entities or ngrams):
        raise ValueError("`entities` and/or `ngrams` must be included")
    if not (entities is None or isinstance(entities, bool)):
        raise ValueError(
            "entities={} is invalid; choices are {}".format(
                entities,
                {True, False, None},
            )
        )
    if not (normalize in ("lemma", "lower") or callable(normalize) or not normalize):
        raise ValueError(
            "normalize={} is invalid; choices are {}".format(
                normalize,
                {"lemma", "lower", types.FunctionType, None},
            )
        )
    if ngrams:
        unigrams_ = []
        ngrams_ = []
        ng_kwargs = {
            "filter_stops", "filter_punct", "filter_nums",
            "include_pos", "exclude_pos",
            "min_freq",
        }
        ng_kwargs = {key: val for key, val in kwargs.items() if key in ng_kwargs}
        for n in sorted(utils.to_collection(ngrams, int, set)):
            # use a faster function for unigrams
            if n == 1:
                unigrams_ = extract.words(doc, **ng_kwargs)
            else:
                ngrams_.append(extract.ngrams(doc, n, **ng_kwargs))
        ngrams_ = itertoolz.concat(ngrams_)
    if entities is not None:
        ent_kwargs = {"include_types", "exclude_types", "drop_determiners", "min_freq"}
        ent_kwargs = {key: val for key, val in kwargs.items() if key in ent_kwargs}
        entities_ = extract.entities(doc, **ent_kwargs)
    if ngrams:
        # use ngrams as-is
        if entities is None:
            terms = itertoolz.concatv(unigrams_, ngrams_)
        # remove unigrams + ngrams that are duplicates of entities
        else:
            entities_ = tuple(entities_)
            ent_idxs = {(ent.start, ent.end) for ent in entities_}
            unigrams_ = (
                ug
                for ug in unigrams_
                if (ug.i, ug.i + 1) not in ent_idxs
            )
            ngrams_ = (
                ng
                for ng in ngrams_
                if (ng.start, ng.end) not in ent_idxs
            )
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
    doc,
    ngrams=(1, 2, 3),
    entities=True,
    normalize="lemma",
    weighting="count",
    as_strings=False,
    **kwargs
):
    """
    Transform ``Doc`` into a bag-of-terms: the set of unique terms in ``Doc``
    mapped to their frequency of occurrence, where "terms" includes ngrams and/or entities.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
            (default) includes unigrams (words), bigrams, and trigrams; `2`
            if only bigrams are wanted; falsy (e.g. False) to not include any
        entities (bool): If True (default), include named entities;
            note: if ngrams are also included, any ngrams that exactly
            overlap with an entity are skipped to prevent double-counting
        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if falsy, use the form of terms as they appear
            in ``doc``; if a callable, must accept a ``Token`` or ``Span``
            and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`.
        weighting ({"count", "freq", "binary"}): Type of weight to assign to
            terms. If "count" (default), weights are the absolute number of
            occurrences (count) of term in doc. If "binary", all counts are
            set equal to 1. If "freq", term counts are normalized by the
            total token count, giving their relative frequency of occurrence.
        as_strings (bool): If True, words are returned as strings; if False
            (default), words are returned as their unique integer ids.
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
        dict: mapping of a unique term id or string (depending on the value
        of ``as_strings``) to its absolute, relative, or binary frequency
        of occurrence (depending on the value of ``weighting``).

    See Also:
        :func:`to_terms_list()`, which is used under the hood.
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


def to_bag_of_words(doc, normalize="lemma", weighting="count", as_strings=False, 
                    remove_stop=True, remove_punct=True, remove_space=True):
    """
    Transform ``Doc`` into a bag-of-words: the set of unique words in ``Doc``
    mapped to their absolute, relative, or binary frequency of occurrence.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        normalize (str): If "lemma", lemmatize words before counting; if
            "lower", lowercase words before counting; otherwise, words are
            counted using the form with which they they appear in doc
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
        remove_stop (bool): If True (default), stop words are removed after
            counting.
        remove_punct (bool): If True (default), punctuation tokens are removed
            after counting.
        remove_space (bool): If True (default), whitespace tokens are removed
            after counting.

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
            if not ( (lex.is_stop and remove_stop) or 
                     (lex.is_punct and remove_punct) or 
                     (lex.is_space and remove_space)):
                bow[wid] = weight
    else:
        ss = doc.vocab.strings
        for wid, weight in wid_weights.items():
            lex = vocab[wid]
            if not ( (lex.is_stop and remove_stop) or 
                     (lex.is_punct and remove_punct) or 
                     (lex.is_space and remove_space) ):
                bow[ss[wid]] = weight
    return bow


def to_semantic_network(
    doc,
    nodes="words",
    normalize="lemma",
    edge_weighting="default",
    window_width=10,
):
    """
    Transform ``Doc`` into a semantic network, where nodes are either
    "words" or "sents" and edges between nodes may be weighted in different ways.

    Args:
        doc (:class:`spacy.tokens.Doc`)
        nodes ({"words", "sents"}): Type of doc component to use as nodes
            in the semantic network.
        normalize (str or callable): If "lemma", lemmatize terms; if "lower",
            lowercase terms; if falsy, use the form of terms as they appear
            in doc; if a callable, must accept a ``Token`` or ``Span``
            (if ``nodes`` = "words" or "sents", respectively) and return a str,
            e.g. :func:`get_normalized_text() <textacy.spacier.utils.get_normalized_text>`
        edge_weighting (str): Type of weighting to apply to edges between nodes;
            if ``nodes`` = "words", options are {"cooc_freq", "binary"},
            if ``nodes`` = "sents", options are {"cosine", "jaccard"}; if
            "default", "cooc_freq" or "cosine" will be automatically used.
        window_width (int): Size of sliding window over terms that determines
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
