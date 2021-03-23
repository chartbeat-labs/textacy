from __future__ import annotations

import collections
import itertools
from operator import attrgetter
from typing import Iterable, List, Pattern, Tuple

from spacy.symbols import (
    AUX, CONJ, DET, VERB,
    agent, attr, aux, auxpass, csubj, csubjpass, dobj, neg, nsubj, nsubjpass, obj, pobj, xcomp,
)
from spacy.tokens import Doc, Span, Token

from . import matches


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


def semistructured_statements_v2(
    doclike: Doc | Span,
    *,
    entity: str | Pattern,
    cue: str,
) -> Iterable[Tuple[List[Token], List[Token], List[Token]]]:
    """
    """
    for entity_cand in matches.regex_matches(doclike, entity, expand=False):
        # is the entity candidate a nominal subject?
        if entity_cand.root.dep in _NOMINAL_SUBJ_DEPS:
            cue_cand = entity_cand.root.head
            # is the cue candidate a verb with matching lemma?
            if cue_cand.pos in {VERB, AUX} and cue_cand.lemma_ == cue:
                frag_cand = None
                for tok in cue_cand.children:
                    if (
                        tok.dep in {attr, dobj, obj} or
                        tok.dep_ == "dative" or
                        (tok.dep == xcomp and not any(child.dep == dobj for child in cue_cand.children))
                    ):
                        frag_cand = list(tok.subtree)
                        break
                if frag_cand is not None:
                    yield (
                        list(entity_cand),
                        sorted(expand_verb(cue_cand), key=attrgetter("i")),
                        sorted(frag_cand, key=attrgetter("i")),
                    )


def semistructured_statements(
    doc: Doc,
    entity: str,
    *,
    cue: str = "be",
    ignore_entity_case: bool = True,
    min_n_words: int = 1,
    max_n_words: int = 20,
) -> Tuple[Span | Token, Span | Token, Span]:
    """
    Extract "semi-structured statements" from a document as a sequence of
    (entity, cue, fragment) triples.

    Args:
        doc
        entity: a noun or noun phrase of some sort (e.g. "President Obama",
            "global warming", "Python")
        cue: verb lemma with which ``entity`` is associated
            (e.g. "talk about", "have", "write")
        ignore_entity_case: If True, entity matching is case-independent
        min_n_words: Min number of tokens allowed in a matching fragment
        max_n_words: Max number of tokens allowed in a matching fragment

    Yields:
        Next matching triple, consisting of (entity, cue, fragment).

    Notes:
        Inspired by N. Diakopoulos, A. Zhang, A. Salway. Visual Analytics of
        Media Frames in Online News and Blogs. IEEE InfoVis Workshop on Text
        Visualization. October, 2013.

        Which itself was inspired by by Salway, A.; Kelly, L.; Skadiņa, I.; and
        Jones, G. 2010. Portable Extraction of Partially Structured Facts from
        the Web. In Proc. ICETAL 2010, LNAI 6233, 345-356. Heidelberg, Springer.

    See Also:
        :func:`subject_verb_object_triples()`
    """
    if ignore_entity_case is True:
        entity_toks = entity.lower().split(" ")
        get_tok_text = lambda x: x.lower_  # noqa: E731
    else:
        entity_toks = entity.split(" ")
        get_tok_text = lambda x: x.text  # noqa: E731
    first_entity_tok = entity_toks[0]
    n_entity_toks = len(entity_toks)
    cue = cue.lower()
    cue_toks = cue.split(" ")
    n_cue_toks = len(cue_toks)

    def is_good_last_tok(tok):
        if tok.is_punct:
            return False
        if tok.pos in {CONJ, DET}:
            return False
        return True

    for sent in doc.sents:
        for tok in sent:

            # filter by entity
            if get_tok_text(tok) != first_entity_tok:
                continue
            if n_entity_toks == 1:
                the_entity = tok
                the_entity_root = the_entity
            if tok.i + n_cue_toks >= len(doc):
                continue
            elif all(
                get_tok_text(tok.nbor(i=i + 1)) == et
                for i, et in enumerate(entity_toks[1:])
            ):
                the_entity = doc[tok.i : tok.i + n_entity_toks]
                the_entity_root = the_entity.root
            else:
                continue

            # filter by cue
            terh = the_entity_root.head
            if terh.lemma_ != cue_toks[0]:
                continue
            if n_cue_toks == 1:
                min_cue_i = terh.i
                max_cue_i = terh.i + n_cue_toks
                the_cue = terh
            elif all(
                terh.nbor(i=i + 1).lemma_ == ct for i, ct in enumerate(cue_toks[1:])
            ):
                min_cue_i = terh.i
                max_cue_i = terh.i + n_cue_toks
                the_cue = doc[terh.i : max_cue_i]
            else:
                continue
            if the_entity_root in the_cue.rights:
                continue

            # now add adjacent auxiliary and negating tokens to the cue, for context
            try:
                min_cue_i = min(
                    left.i
                    for left in itertools.takewhile(
                        lambda x: x.dep_ in {"aux", "neg"},
                        reversed(list(the_cue.lefts)),
                    )
                )
            except ValueError:
                pass
            try:
                max_cue_i = max(
                    right.i
                    for right in itertools.takewhile(
                        lambda x: x.dep_ in {"aux", "neg"}, the_cue.rights
                    )
                )
            except ValueError:
                pass
            if max_cue_i - min_cue_i > 1:
                the_cue = doc[min_cue_i:max_cue_i]
            else:
                the_cue = doc[min_cue_i]

            # filter by fragment
            try:
                min_frag_i = min(right.left_edge.i for right in the_cue.rights)
                max_frag_i = max(right.right_edge.i for right in the_cue.rights)
            except ValueError:
                continue
            while is_good_last_tok(doc[max_frag_i]) is False:
                max_frag_i -= 1
            n_fragment_toks = max_frag_i - min_frag_i
            if (
                n_fragment_toks <= 0
                or n_fragment_toks < min_n_words
                or n_fragment_toks > max_n_words
            ):
                continue
            # HACK...
            if min_frag_i == max_cue_i - 1:
                min_frag_i += 1
            the_fragment = doc[min_frag_i : max_frag_i + 1]

            yield (the_entity, the_cue, the_fragment)
