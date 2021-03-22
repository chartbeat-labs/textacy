from __future__ import annotations

import collections
import itertools
from operator import attrgetter
from typing import Iterable, List, Tuple

from spacy.symbols import (
    CONJ, DET, VERB,
    agent, aux, auxpass, csubj, csubjpass, dobj, neg, nsubj, nsubjpass, pobj, xcomp,
)
from spacy.tokens import Doc, Span, Token


_NOMINAL_SUBJ_DEPS = {nsubj, nsubjpass}
_CLAUSAL_SUBJ_DEPS = {csubj, csubjpass}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}


def subject_verb_object_triples(
    doclike: Doc | Span,
) -> Iterable[Tuple[List[Token], List[Token], List[Token]]]:
    """
    Extract an ordered sequence of (subject, verb, object) triples from a document
    or sentence.

    Args:
        doclike

    Yields:
        Next SVO triple, in (more or less) order of appearance.

    See Also:
        :func:`semistructured_statements()`
    """
    if isinstance(doclike, Span):
        sents = [doclike]
    else:
        sents = doclike.sents

    for sent in sents:
        # connect subjects/objects to direct verb heads
        # and expand them to include conjuncts, compound nouns, ...
        verb_sos = collections.defaultdict(lambda: collections.defaultdict(set))
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
                if (
                    head.pos == VERB and
                    not any(child.dep == dobj for child in head.children)
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
                yield (
                    sorted(so_dict["subjects"], key=attrgetter("i")),
                    sorted(expand_verb(verb), key=attrgetter("i")),
                    sorted(so_dict["objects"], key=attrgetter("i")),
                )


def expand_noun(tok: Token) -> List[Token]:
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds


def expand_verb(tok: Token) -> List[Token]:
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers
