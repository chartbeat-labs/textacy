from __future__ import annotations

import collections
from operator import attrgetter
from typing import Iterable, List, Tuple

from spacy.symbols import agent, aux, auxpass, dobj, neg, nsubj, nsubjpass, pobj, VERB
from spacy.tokens import Doc, Span, Token


_NOMINAL_SUBJ_DEPS = {nsubj, nsubjpass}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}


def subject_verb_object_triples(
    doclike: Doc | Span,
) -> Iterable[Tuple[List[Token], List[Token], List[Token]]]:
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a document
    or sentence.

    Args:
        doclike

    Yields:
        Next SVO triple, in (more or less) order of appearance.
    """
    if isinstance(doclike, Span):
        sents = [doclike]
    else:
        sents = doclike.sents

    verb_sos = collections.defaultdict(lambda: collections.defaultdict(list))
    # connect subjects/objects to direct verb heads
    # and expand them to include conjuncts, compound nouns, ...
    for sent in sents:
        for tok in sent:
            head = tok.head
            if tok.dep in _NOMINAL_SUBJ_DEPS:
                if head.pos == VERB:
                    verb = head
                    verb_sos[verb]["subjects"].extend(expand_noun(tok))
            elif tok.dep == dobj:
                if head.pos == VERB:
                    verb = head
                    verb_sos[verb]["objects"].extend(expand_noun(tok))
            elif tok.dep == pobj:
                if head.dep == agent and head.head.pos == VERB:
                    verb = head.head
                    verb_sos[verb]["objects"].extend(expand_noun(tok))
            # TODO: handle clausal subjects? (ccomp)
    # fill in any indirect relationships connected via verb conjuncts
    for verb, so_dict in verb_sos.items():
        conjuncts = verb.conjuncts
        if so_dict.get("subjects"):
            for conj in conjuncts:
                conj_so_dict = verb_sos.get(conj)
                if conj_so_dict and not conj_so_dict.get("subjects"):
                    conj_so_dict["subjects"].extend(so_dict["subjects"])
        if not so_dict.get("objects"):
            so_dict["objects"].extend(
                obj
                for conj in conjuncts
                for obj in verb_sos.get(conj, {}).get("objects", [])
            )
    # expand verbs and restructure into svo triples
    for verb, so_dict in verb_sos.items():
        if so_dict["subjects"] and so_dict["objects"]:
            yield (so_dict["subjects"], expand_verb(verb), so_dict["objects"])


def expand_noun(tok: Token) -> List[Token]:
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        # TODO: why doesn't compound import from spacy.symbols?
        if child.dep_ == "compound"
    ]
    return sorted(tok_and_conjuncts + compounds, key=attrgetter("i"))


def expand_verb(tok: Token) -> List[Token]:
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return sorted([tok] + verb_modifiers, key=attrgetter("i"))
