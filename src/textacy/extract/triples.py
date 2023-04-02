"""
Triples
-------

:mod:`textacy.extract.triples`: Extract structured triples from a document or sentence
through rule-based pattern-matching of the annotated tokens.
"""
from __future__ import annotations

import collections
from operator import attrgetter
from typing import Iterable, Mapping, Optional, Pattern

from cytoolz import itertoolz
from spacy.symbols import (
    AUX,
    VERB,
    agent,
    attr,
    aux,
    auxpass,
    csubj,
    csubjpass,
    dobj,
    neg,
    nsubj,
    nsubjpass,
    obj,
    pobj,
    xcomp,
)
from spacy.tokens import Doc, Span, Token

from .. import constants, types, utils
from . import matches


_NOMINAL_SUBJ_DEPS = {nsubj, nsubjpass}
_CLAUSAL_SUBJ_DEPS = {csubj, csubjpass}
_ACTIVE_SUBJ_DEPS = {csubj, nsubj}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}

SVOTriple: tuple[list[Token], list[Token], list[Token]] = collections.namedtuple(
    "SVOTriple", ["subject", "verb", "object"]
)
SSSTriple: tuple[list[Token], list[Token], list[Token]] = collections.namedtuple(
    "SSSTriple", ["entity", "cue", "fragment"]
)
DQTriple: tuple[list[Token], list[Token], Span] = collections.namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)


def subject_verb_object_triples(doclike: types.DocLike) -> Iterable[SVOTriple]:
    """
    Extract an ordered sequence of subject-verb-object triples from a document
    or sentence.

    Args:
        doclike

    Yields:
        Next SVO triple as (subject, verb, object), in approximate order of appearance.
    """
    sents: Iterable[Span]
    if isinstance(doclike, Span):
        sents = [doclike]
    else:
        sents = doclike.sents

    for sent in sents:
        # connect subjects/objects to direct verb heads
        # and expand them to include conjuncts, compound nouns, ...
        verb_sos: Mapping = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for tok in sent:
            head = tok.head
            # ensure entry for all verbs, even if empty
            # to catch conjugate verbs without direct subject/object deps
            if tok.pos == VERB:
                _ = verb_sos[tok]
            # nominal subject of active or passive verb
            if tok.dep in _NOMINAL_SUBJ_DEPS:
                if head.pos == VERB:
                    verb_sos[head]["subjects"].update(expand_noun(tok))
            # clausal subject of active or passive verb
            elif tok.dep in _CLAUSAL_SUBJ_DEPS:
                if head.pos == VERB:
                    verb_sos[head]["subjects"].update(tok.subtree)
            # nominal direct object of transitive verb
            elif tok.dep == dobj:
                if head.pos == VERB:
                    verb_sos[head]["objects"].update(expand_noun(tok))
            # prepositional object acting as agent of passive verb
            elif tok.dep == pobj:
                if head.dep == agent and head.head.pos == VERB:
                    verb_sos[head.head]["objects"].update(expand_noun(tok))
            # open clausal complement, but not as a secondary predicate
            elif tok.dep == xcomp:
                if head.pos == VERB and not any(
                    child.dep == dobj for child in head.children
                ):
                    # TODO: just the verb, or the whole tree?
                    # verb_sos[verb]["objects"].update(expand_verb(tok))
                    verb_sos[head]["objects"].update(tok.subtree)
        # fill in any indirect relationships connected via verb conjuncts
        for verb, so_dict in verb_sos.items():
            conjuncts = verb.conjuncts
            if so_dict.get("subjects"):
                for conj in conjuncts:
                    conj_so_dict = verb_sos.get(conj)
                    if conj_so_dict and not conj_so_dict.get("subjects"):
                        conj_so_dict["subjects"].update(so_dict["subjects"])
            if not so_dict.get("objects"):
                so_dict["objects"].update(
                    obj
                    for conj in conjuncts
                    for obj in verb_sos.get(conj, {}).get("objects", [])
                )
        # expand verbs and restructure into svo triples
        for verb, so_dict in verb_sos.items():
            if so_dict["subjects"] and so_dict["objects"]:
                yield SVOTriple(
                    subject=sorted(so_dict["subjects"], key=attrgetter("i")),
                    verb=sorted(expand_verb(verb), key=attrgetter("i")),
                    object=sorted(so_dict["objects"], key=attrgetter("i")),
                )


def semistructured_statements(
    doclike: types.DocLike,
    *,
    entity: str | Pattern,
    cue: str,
    fragment_len_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> Iterable[SSSTriple]:
    """
    Extract "semi-structured statements" from a document as a sequence of
    (entity, cue, fragment) triples.

    Args:
        doclike
        entity: Noun or noun phrase of interest expressed as a regular expression
            pattern string (e.g. ``"[Gg]lobal [Ww]arming"``) or compiled object
            (e.g. ``re.compile("global warming", re.IGNORECASE)``).
        cue: Verb lemma with which ``entity`` is associated (e.g. "be", "have", "say").
        fragment_len_range: Filter statements to those whose fragment length in tokens
            is within the specified [low, high) interval. Both low and high values
            must be specified, but a null value for either is automatically replaced
            by safe default values. None (default) skips filtering by fragment length.

    Yields:
        Next matching triple, consisting of (entity, cue, fragment),
        in order of appearance.

    Notes:
        Inspired by N. Diakopoulos, A. Zhang, A. Salway. Visual Analytics of
        Media Frames in Online News and Blogs. IEEE InfoVis Workshop on Text
        Visualization. October, 2013.

        Which itself was inspired by by Salway, A.; Kelly, L.; Skadiņa, I.; and
        Jones, G. 2010. Portable Extraction of Partially Structured Facts from
        the Web. In Proc. ICETAL 2010, LNAI 6233, 345-356. Heidelberg, Springer.
    """
    if fragment_len_range is not None:
        fragment_len_range = utils.validate_and_clip_range(
            fragment_len_range, (1, 1000), int
        )
    for entity_cand in matches.regex_matches(doclike, entity, alignment_mode="strict"):
        # is the entity candidate a nominal subject?
        if entity_cand.root.dep in _NOMINAL_SUBJ_DEPS:
            cue_cand = entity_cand.root.head
            # is the cue candidate a verb with matching lemma?
            if cue_cand.pos in {VERB, AUX} and cue_cand.lemma_ == cue:
                frag_cand = None
                for tok in cue_cand.children:
                    if (
                        tok.dep in {attr, dobj, obj}
                        or tok.dep_ == "dative"
                        or (
                            tok.dep == xcomp
                            and not any(
                                child.dep == dobj for child in cue_cand.children
                            )
                        )
                    ):
                        subtoks = list(tok.subtree)
                        if (
                            fragment_len_range is None
                            or fragment_len_range[0]
                            <= len(subtoks)
                            < fragment_len_range[1]
                        ):
                            frag_cand = subtoks
                            break
                if frag_cand is not None:
                    yield SSSTriple(
                        entity=list(entity_cand),
                        cue=sorted(expand_verb(cue_cand), key=attrgetter("i")),
                        fragment=sorted(frag_cand, key=attrgetter("i")),
                    )


def direct_quotations(doc: Doc) -> Iterable[DQTriple]:
    """
    Extract direct quotations with an attributable speaker from a document
    using simple rules and patterns. Does not extract indirect or mixed quotations!

    Args:
        doc

    Yields:
        Next direct quotation in ``doc`` as a (speaker, cue, content) triple.

    Notes:
        Loosely inspired by Krestel, Bergler, Witte. "Minding the Source: Automatic
        Tagging of Reported Speech in Newspaper Articles".
    """
    # TODO: train a model to do this instead, maybe similar to entity recognition
    try:
        _reporting_verbs = constants.REPORTING_VERBS[doc.lang_]
    except KeyError:
        raise ValueError(
            f"direct quotation extraction is not implemented for lang='{doc.lang_}', "
            f"only {sorted(constants.REPORTING_VERBS.keys())}"
        )
    qtok_idxs = [tok.i for tok in doc if tok.is_quote]
    if len(qtok_idxs) % 2 != 0:
        raise ValueError(
            f"{len(qtok_idxs)} quotation marks found, indicating an unclosed quotation; "
            "given the limitations of this method, it's safest to bail out "
            "rather than guess which quotation is unclosed"
        )
    qtok_pair_idxs = list(itertoolz.partition(2, qtok_idxs))
    for qtok_start_idx, qtok_end_idx in qtok_pair_idxs:
        content = doc[qtok_start_idx : qtok_end_idx + 1]
        cue = None
        speaker = None
        # filter quotations by content
        if (
            # quotations should have at least a couple tokens
            # excluding the first/last quotation mark tokens
            len(content) < 4
            # filter out titles of books and such, if possible
            or all(
                tok.is_title
                for tok in content
                # if tok.pos in {NOUN, PROPN}
                if not (tok.is_punct or tok.is_stop)
            )
            # TODO: require closing punctuation before the quotation mark?
            # content[-2].is_punct is False
        ):
            continue
        # get window of adjacent/overlapping sentences
        window_sents = (
            sent
            for sent in doc.sents
            # these boundary cases are a subtle bit of work...
            if (
                (sent.start < qtok_start_idx and sent.end >= qtok_start_idx - 1)
                or (sent.start <= qtok_end_idx + 1 and sent.end > qtok_end_idx)
            )
        )
        # get candidate cue verbs in window
        cue_cands = [
            tok
            for sent in window_sents
            for tok in sent
            if (
                tok.pos == VERB
                and tok.lemma_ in _reporting_verbs
                # cue verbs must occur *outside* any quotation content
                and not any(
                    qts_idx <= tok.i <= qte_idx for qts_idx, qte_idx in qtok_pair_idxs
                )
            )
        ]
        # sort candidates by proximity to quote content
        cue_cands = sorted(
            cue_cands,
            key=lambda cc: min(abs(cc.i - qtok_start_idx), abs(cc.i - qtok_end_idx)),
        )
        for cue_cand in cue_cands:
            if cue is not None:
                break
            for speaker_cand in cue_cand.children:
                if speaker_cand.dep in _ACTIVE_SUBJ_DEPS:
                    cue = expand_verb(cue_cand)
                    speaker = expand_noun(speaker_cand)
                    break
        if content and cue and speaker:
            yield DQTriple(
                speaker=sorted(speaker, key=attrgetter("i")),
                cue=sorted(cue, key=attrgetter("i")),
                content=content,
            )


def expand_noun(tok: Token) -> list[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds


def expand_verb(tok: Token) -> list[Token]:
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers
