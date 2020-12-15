"""
:mod:`textacy.spacier.utils`: Helper functions for working with / extending spaCy's
core functionality.
"""
import itertools
from typing import Iterable, List, Tuple, Union

import numpy as np
from spacy import attrs
from spacy.language import Language
from spacy.symbols import PROPN, VERB, NOUN
from spacy.tokens import Doc, Span, Token

from .. import constants, errors, text_utils
from . import core


def make_doc_from_text_chunks(
    text: str, lang: Union[str, Language], chunk_size: int = 100000,
) -> Doc:
    """
    Make a single spaCy-processed document from 1 or more chunks of ``text``.
    This is a workaround for processing very long texts, for which spaCy
    is unable to allocate enough RAM.

    Although this function's performance is *pretty good*, it's inherently
    less performant that just processing the entire text in one shot.
    Only use it if necessary!

    Args:
        text: Text document to be chunked and processed by spaCy.
        lang: A 2-letter language code (e.g. "en"),
            the name of a spaCy model for the desired language, or
            an already-instantiated spaCy language pipeline.
        chunk_size: Number of characters comprising each text chunk
            (excluding the last chunk, which is probably smaller).
            For best performance, value should be somewhere between 1e3 and 1e7,
            depending on how much RAM you have available.

            .. note:: Since chunking is done by character, chunks edges' probably
               won't respect natural language segmentation, which means that every
               ``chunk_size`` characters, spaCy will probably get tripped up and
               make weird parsing errors.

    Returns:
        A single processed document, initialized from components accumulated chunk by chunk.
    """
    if isinstance(lang, str):
        lang = core.load_spacy_lang(lang)
    elif not isinstance(lang, Language):
        raise TypeError(
            errors.type_invalid_msg("lang", type(lang), Union[str, Language])
        )

    words: List[str] = []
    spaces: List[bool] = []
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
    doc = Doc(lang.vocab, words=words, spaces=spaces)
    doc = doc.from_array(cols, np.concatenate(np_arrays, axis=0))

    return doc


def merge_spans(spans: Iterable[Span], doc: Doc) -> None:
    """
    Merge spans into single tokens in ``doc``, *in-place*.

    Args:
        spans (Iterable[:class:`spacy.tokens.Span`])
        doc (:class:`spacy.tokens.Doc`)
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


def preserve_case(token: Token) -> bool:
    """
    Return True if ``token`` is a proper noun or acronym; otherwise, False.

    Raises:
        ValueError: If parent document has not been POS-tagged.
    """
    if token.doc.is_tagged is False:
        raise ValueError(f"parent doc of token '{token}' has not been POS-tagged")
    if token.pos == PROPN or text_utils.is_acronym(token.text):
        return True
    else:
        return False


def get_normalized_text(span_or_token: Union[Span, Token]) -> str:
    """
    Get the text of a spaCy span or token, normalized depending on its characteristics.
    For proper nouns and acronyms, text is returned as-is; for everything else,
    text is lemmatized.
    """
    if isinstance(span_or_token, Token):
        return (
            span_or_token.text if preserve_case(span_or_token) else span_or_token.lemma_
        )
    elif isinstance(span_or_token, Span):
        return " ".join(
            token.text if preserve_case(token) else token.lemma_
            for token in span_or_token
        )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "span_or_token", type(span_or_token), Union[Span, Token]
            )
        )


def get_main_verbs_of_sent(sent: Span) -> List[Token]:
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in constants.AUX_DEPS
    ]


def get_subjects_of_verb(verb: Token) -> List[Token]:
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in constants.SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb: Token) -> List[Token]:
    """
    Return all objects of a verb according to the dependency parse,
    including open clausal complements.
    """
    objs = [tok for tok in verb.rights if tok.dep_ in constants.OBJ_DEPS]
    # get open clausal complements (xcomp)
    objs.extend(tok for tok in verb.rights if tok.dep_ == "xcomp")
    # get additional conjunct objects
    objs.extend(tok for obj in objs for tok in _get_conjuncts(obj))
    # get additional prepositional objects
    for tok in verb.rights:
        if tok.dep_ == "prep":
            for tok2 in tok.rights:
                if tok2.pos in [PROPN, NOUN]:  # pragma: no cover
                    objs.append(tok2)
    return objs


def _get_conjuncts(tok: Token) -> List[Token]:
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun: Token) -> Tuple[int, int]:
    """Return document indexes spanning all (adjacent) tokens in a compound noun."""
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb: Token) -> Tuple[int, int]:
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
        for _ in itertools.takewhile(lambda x: x.dep_ in constants.AUX_DEPS, verb.rights)
    )
    return (min_i, max_i)
