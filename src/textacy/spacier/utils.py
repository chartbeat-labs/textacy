"""
spaCy Utils
-----------

:mod:`textacy.spacier.utils`: Helper functions for working with / extending spaCy's
core functionality.
"""
from __future__ import annotations

import functools
import itertools
import pathlib
from typing import Iterable, Union

from cachetools import cached
from cachetools.keys import hashkey
from spacy import attrs
from spacy.language import Language
from spacy.morphology import Morphology
from spacy.pipeline import Morphologizer
from spacy.symbols import PROPN, VERB
from spacy.tokens import Doc, Span, Token

from .. import cache, constants, errors, types, utils
from . import core


def make_doc_from_text_chunks(
    text: str, lang: types.LangLike, chunk_size: int = 100000
) -> Doc:
    """
    Make a single spaCy-processed document from 1 or more chunks of ``text``.
    This is a workaround for processing very long texts, for which spaCy
    is unable to allocate enough RAM.

    Args:
        text: Text document to be chunked and processed by spaCy.
        lang: Language with which spaCy processes ``text``, represented as
            the full name of or path on disk to the pipeline, or
            an already instantiated pipeline instance.
        chunk_size: Number of characters comprising each text chunk
            (excluding the last chunk, which is probably smaller).
            For best performance, value should be somewhere between 1e3 and 1e7,
            depending on how much RAM you have available.

            .. note:: Since chunking is done by character, chunks edges' probably
               won't respect natural language segmentation, which means that every
               ``chunk_size`` characters, spaCy's models may make mistakes.

    Returns:
        A single processed document, built from concatenated text chunks.
    """
    utils.deprecated(
        "This function is deprecated, and will be removed in a future version. "
        "Instead, use the usual :func:`textacy.make_spacy_doc()` "
        "and specify a non-null `chunk_size`",
        action="once",
    )
    lang = resolve_langlike(lang)
    text_chunks = (text[i : i + chunk_size] for i in range(0, len(text), chunk_size))
    docs = list(lang.pipe(text_chunks))
    return Doc.from_docs(docs)


def merge_spans(spans: Iterable[Span], doc: Doc) -> None:
    """
    Merge spans into single tokens in ``doc``, *in-place*.

    Args:
        spans (Iterable[:class:`spacy.tokens.Span`])
        doc (:class:`spacy.tokens.Doc`)
    """
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            retokenizer.merge(
                doc[span.start : span.end],
                attrs=attrs.intify_attrs({"ent_type": span.label}, string_store),
            )


def preserve_case(token: Token) -> bool:
    """
    Return True if ``token`` is a proper noun or acronym; otherwise, False.

    Raises:
        ValueError: If parent document has not been POS-tagged.
    """
    if token.doc.has_annotation("TAG") is False:
        raise ValueError(f"parent doc of token '{token}' has not been POS-tagged")
    # is_acronym() got moved into a subpkg with heavier dependencies that we don't
    # want imported at the top-level package; this is the only outside place
    # that uses that function, so let's hide the required import in this function
    # using a try/except for performance
    # not the prettiest solution, but should be alright
    try:
        acros.is_acronym
    except NameError:
        from textacy.extract import acros
    if token.pos == PROPN or acros.is_acronym(token.text):
        return True
    else:
        return False


def get_normalized_text(span_or_token: Span | Token) -> str:
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


def get_main_verbs_of_sent(sent: Span) -> list[Token]:
    """Return the main (non-auxiliary) verbs in a sentence."""
    return [
        tok for tok in sent if tok.pos == VERB and tok.dep_ not in constants.AUX_DEPS
    ]


def get_subjects_of_verb(verb: Token) -> list[Token]:
    """Return all subjects of a verb according to the dependency parse."""
    subjs = [tok for tok in verb.lefts if tok.dep_ in constants.SUBJ_DEPS]
    # get additional conjunct subjects
    subjs.extend(tok for subj in subjs for tok in _get_conjuncts(subj))
    return subjs


def get_objects_of_verb(verb: Token) -> list[Token]:
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


def _get_conjuncts(tok: Token) -> list[Token]:
    """
    Return conjunct dependents of the leftmost conjunct in a coordinated phrase,
    e.g. "Burton, [Dan], and [Josh] ...".
    """
    return [right for right in tok.rights if right.dep_ == "conj"]


def get_span_for_compound_noun(noun: Token) -> tuple[int, int]:
    """Return document indexes spanning all (adjacent) tokens in a compound noun."""
    min_i = noun.i - sum(
        1
        for _ in itertools.takewhile(
            lambda x: x.dep_ == "compound", reversed(list(noun.lefts))
        )
    )
    return (min_i, noun.i)


def get_span_for_verb_auxiliaries(verb: Token) -> tuple[int, int]:
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


def resolve_langlike(lang: types.LangLike) -> Language:
    if isinstance(lang, Language):
        return lang
    elif isinstance(lang, (str, pathlib.Path)):
        return core.load_spacy_lang(lang)
    else:
        raise TypeError(errors.type_invalid_msg("lang", type(lang), types.LangLike))


def resolve_langlikeincontext(text: str, lang: types.LangLikeInContext) -> Language:
    if isinstance(lang, Language):
        return lang
    elif isinstance(lang, (str, pathlib.Path)):
        return core.load_spacy_lang(lang)
    elif callable(lang):
        return resolve_langlikeincontext(text, lang(text))
    else:
        raise TypeError(
            errors.type_invalid_msg("lang", type(lang), types.LangLikeInContext)
        )


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "spacy_lang_morph_labels"))
def get_spacy_lang_morph_labels(lang: types.LangLike) -> set[str]:
    """
    Get the full set of morphological feature labels assigned
    by a spaCy language pipeline according to its "morphologizer" pipe's metadata,
    or just get the default set of Universal Dependencies (v2) feature labels.

    Args:
        lang: Language with which spaCy processes text, represented as the full name
            of a spaCy language pipeline, the path on disk to it,
            or an already instantiated pipeline.

    Returns:
        Set of morphological feature labels assigned/assignable by ``lang``.
    """
    spacy_lang = resolve_langlike(lang)
    if spacy_lang.has_pipe("morphologizer"):
        morphologizer = spacy_lang.get_pipe("morphologizer")
    elif any(isinstance(comp, Morphologizer) for _, comp in spacy_lang.pipeline):
        for _, component in spacy_lang.pipeline:
            if isinstance(component, Morphologizer):
                morphologizer = component
                break
        else:
            return constants.UD_V2_MORPH_LABELS  # mypy not smart enough to know better
    else:
        return constants.UD_V2_MORPH_LABELS

    assert isinstance(morphologizer, Morphologizer)  # type guard
    return {
        feat_name
        for label in morphologizer.labels
        for feat_name in Morphology.feats_to_dict(label).keys()
    }
