"""
spaCy Utils
-----------

Helper functions for working with / extending spaCy's core functionality.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

import numpy as np
from spacy import attrs
from spacy.language import Language as SpacyLang
from spacy.symbols import NOUN, PROPN, VERB
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.span import Span as SpacySpan
from spacy.tokens.token import Token as SpacyToken

from .. import cache
from .. import compat
from .. import constants
from .. import text_utils


def make_doc_from_text_chunks(text, lang, chunk_size=100000):
    """
    Make a single spaCy-processed document from 1 or more chunks of ``text``.
    This is a workaround for processing very long texts, for which spaCy
    is unable to allocate enough RAM.

    Although this function's performance is *pretty good*, it's inherently
    less performant that just processing the entire text in one shot.
    Only use it if necessary!

    Args:
        text (str): Text document to be chunked and processed by spaCy.
        lang (str or ``spacy.Language``): A 2-letter language code (e.g. "en"),
            the name of a spaCy model for the desired language, or
            an already-instantiated spaCy language pipeline.
        chunk_size (int): Number of characters comprising each text chunk
            (excluding the last chunk, which is probably smaller). For best
            performance, value should be somewhere between 1e3 and 1e7,
            depending on how much RAM you have available.

            .. note:: Since chunking is done by character, chunks edges' probably
               won't respect natural language segmentation, which means that every
               ``chunk_size`` characters, spaCy will probably get tripped up and
               make weird parsing errors.

    Returns:
        ``spacy.Doc``: A single processed document, initialized from
        components accumulated chunk by chunk.
    """
    if isinstance(lang, compat.unicode_):
        lang = cache.load_spacy(lang)
    elif not isinstance(lang, SpacyLang):
        raise TypeError(
            "`lang` must be {}, not {}".format({compat.unicode_, SpacyLang}, type(lang))
        )

    words = []
    spaces = []
    np_arrays = []
    cols = [attrs.POS, attrs.TAG, attrs.DEP, attrs.HEAD, attrs.ENT_IOB, attrs.ENT_TYPE]
    text_len = len(text)
    i = 0
    # iterate over text chunks and accumulate components needed to make a doc
    while i < text_len:
        chunk_doc = lang(text[i : i + chunk_size])
        words.extend(tok.text for tok in chunk_doc)
        spaces.extend(bool(tok.whitespace_) for tok in chunk_doc)
        np_arrays.append(chunk_doc.to_array(cols))
        i += chunk_size
    # now, initialize the doc from words and spaces
    # then load attribute values from the concatenated np array
    doc = SpacyDoc(lang.vocab, words=words, spaces=spaces)
    doc = doc.from_array(cols, np.concatenate(np_arrays, axis=0))

    return doc


def merge_spans(spans, doc):
    """
    Merge spans into single tokens in ``doc``, *in-place*.

    Args:
        spans (Iterable[``spacy.Span``])
        doc (``spacy.Doc``)
    """
    try:  # retokenizer was added to spacy in v2.0.11
        with doc.retokenize() as retokenizer:
            string_store = doc.vocab.strings
            for span in spans:
                retokenizer.merge(
                    doc[span.start : span.end],
                    attrs=attrs.intify_attrs({"ent_type": span.label}, string_store),
                )
    except AttributeError:
        spans = [(span.start_char, span.end_char, span.label) for span in spans]
        for start_char, end_char, label in spans:
            doc.merge(start_char, end_char, ent_type=label)


def preserve_case(token):
    """
    Return True if ``token`` is a proper noun or acronym; otherwise, False.

    Args:
        token (``spacy.Token``)

    Returns:
        bool

    Raises:
        ValueError: If parent document has not been POS-tagged.
    """
    if token.doc.is_tagged is False:
        raise ValueError(
            'parent doc of token "{}" has not been POS-tagged'.format(token)
        )
    if token.pos == PROPN or text_utils.is_acronym(token.text):
        return True
    else:
        return False


def get_normalized_text(span_or_token):
    """
    Get the text of a spaCy span or token, normalized depending on its characteristics.
    For proper nouns and acronyms, text is returned as-is; for everything else,
    text is lemmatized.

    Args:
        span_or_token (``spacy.Span`` or ``spacy.Token``)

    Returns:
        str
    """
    if isinstance(span_or_token, SpacyToken):
        return (
            span_or_token.text if preserve_case(span_or_token) else span_or_token.lemma_
        )
    elif isinstance(span_or_token, SpacySpan):
        return " ".join(
            token.text if preserve_case(token) else token.lemma_
            for token in span_or_token
        )
    else:
        raise TypeError(
            'input must be a spaCy Token or Span, not "{}"'.format(type(span_or_token))
        )


def get_main_verbs_of_sent(sent):
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in constants.AUX_DEPS
    ]


def get_subjects_of_verb(verb):
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in constants.SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb):
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in constants.OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    return objs


def _get_conjuncts(tok):
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun):
    """
    Return document indexes spanning all (adjacent) tokens
    in a compound noun.
    """
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb):
    """
    Return document indexes spanning all (adjacent) tokens
    around a verb that are auxiliary verbs or negations.
    """
    min_i = verb.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in constants.AUX_DEPS, reversed(list(verb.lefts))
        )
    )
    max_i = verb.i + sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ in constants.AUX_DEPS, verb.rights
        )
    )
    return (min_i, max_i)
