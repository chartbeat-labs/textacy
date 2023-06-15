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

from spacy.symbols import (
    AUX,
    VERB,
    PUNCT,
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
    xcomp
)
from spacy.tokens import Doc, Span, Token
import regex as re

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
        min_quote_length - minimum distance (in tokens) between potentially paired quotation marks.

    Yields:
        Next direct quotation in ``doc`` as a (speaker, cue, content) triple.

    Notes:
        Loosely inspired by Krestel, Bergler, Witte. "Minding the Source: Automatic
        Tagging of Reported Speech in Newspaper Articles".
    """
    try:
        _reporting_verbs = constants.REPORTING_VERBS[doc.lang_]
    except KeyError:
        raise ValueError(
            f"direct quotation extraction is not implemented for lang='{doc.lang_}', "
            f"only {sorted(constants.REPORTING_VERBS.keys())}"
        )
    # pairs up quotation-like characters based on acceptable start/end combos
    # see constants for more info
    qtoks = [tok for tok in doc if tok.is_quote or (tok.is_space and tok.text == "\n")]
    qtok_idx_pairs = [(-1,-1)]
    for n, q in enumerate(qtoks):
        if (
            not bool(q.whitespace_)
            and q.i not in [q_[1] for q_ in qtok_idx_pairs] 
            and q.i > qtok_idx_pairs[-1][1]
            ):
            for q_ in qtoks[n+1:]:
                if (ord(q.text), ord(q_.text)) in constants.QUOTATION_MARK_PAIRS:
                    qtok_idx_pairs.append((q.i, q_.i))
                    break
    
    def filter_quote_tokens(tok):
        return any(qts_idx <= tok.i <= qte_idx for qts_idx, qte_idx in qtok_idx_pairs)

    for qtok_start_idx, qtok_end_idx in qtok_idx_pairs:
        content = doc[qtok_start_idx : qtok_end_idx + 1]
        cue = None
        speaker = None
        # filter quotations by content
        if (
            # quotations should have at least a couple tokens
            # excluding the first/last quotation mark tokens
            len(content) < constants.MIN_QUOTE_LENGTH
            # filter out titles of books and such, if possible
            or all(
                tok.is_title
                for tok in content
                # if tok.pos in {NOUN, PROPN}
                if not (tok.is_punct or tok.is_stop)
            )
        ):
            continue

        for window_sents in [
            windower(qtok_start_idx, qtok_end_idx, doc, True), 
            windower(qtok_start_idx, qtok_end_idx, doc)
        ]:
        # get candidate cue verbs in window
            cue_candidates = [
                    tok
                    for sent in window_sents
                    for tok in sent
                    if tok.pos == VERB 
                    and tok.lemma_ in _reporting_verbs
                    and not filter_quote_tokens(tok)
                ]
            cue_candidates = sorted(cue_candidates,
                key=lambda cc: min(abs(cc.i - qtok_start_idx), abs(cc.i - qtok_end_idx))
            )
            for cue_cand in cue_candidates:
                if cue is not None:
                    break
                speaker_cands = [
                    speaker_cand for speaker_cand in cue_cand.children
                    if speaker_cand.pos!=PUNCT
                    and not filter_quote_tokens(speaker_cand)
                    and ((speaker_cand.i >= qtok_end_idx) 
                        or (speaker_cand.i <= qtok_start_idx ))
                ]
                for speaker_cand in speaker_cands:
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
                break

def expand_noun(tok: Token) -> list[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds

def expand_verb(tok: Token) -> list[Token]:
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers
    
def windower(i, j, doc, by_linebreak: bool=False) -> Iterable:
    """
    Two ways to search for cue and speaker: the old way, and a new way based on line breaks.
    """
    if by_linebreak:
        i_, j_ = line_break_window(i, j, doc)
        if i_ is not None:
            return (sent for sent in doc[i_+1:j_-1].sents)
        else:
            return []
    else:
        # get window of adjacent/overlapping sentences
        return (
            sent
            for sent in doc.sents
            # these boundary cases are a subtle bit of work...
            if (
                (sent.start < i and sent.end >= i - 1)
                or (sent.start <= j + 1 and sent.end > j)
            )
        )

def line_break_window(i, j, doc):
    """
    Finds the boundaries of the paragraph containing doc[i:j].
    """
    lb_tok_idxs = [tok.i for tok in doc if tok.text == "\n"]
    for i_, j_ in zip(lb_tok_idxs, lb_tok_idxs[1:]):
        if i_ <= i and j_ >= j:
            return (i_, j_)
    else:
        return (None, None)
    
def prep_text_for_quote_detection(t: str, fix_plural_possessives: bool=True) -> str:
    """
    Sorts out some common issues that trip up the quote detector. Works best one paragraph at a time -- use prep_document_for_quote_detection for the whole doc.

    - replaces consecutive apostrophes with a double quote (no idea why this happens but it does)
    - adds spaces before or after double quotes that don't have them
    - if enabled, fixes plural possessives by adding an "x", because the hanging apostrophe can trigger quote detection. 
    - adds a double quote to the end of paragraphs that are continuations of quotes and thus traditionally don't end with quotation marks

    Input:
        t (str) - text to be prepped, preferably one paragraph
        fix_plural_possessives (bool) - enables fix_plural_possessives
    
    Output:
        t (str) - text prepped for quote detection
    """
    if not t:
        return
    
    t = t.replace("\'\'", "\"")
    if fix_plural_possessives:
        t = re.sub(r"(.{3,8}s\')(\s)", r"\1x\2", t)
    while re.search(constants.DOUBLE_QUOTES_NOSPACE_REGEX, p):
        match = re.search(constants.DOUBLE_QUOTES_NOSPACE_REGEX, p)
        if len(re.findall(constants.ANY_DOUBLE_QUOTE_REGEX, p[:match.start()])) % 2 != 0:
            replacer = '" '
        else:
            replacer = ' "'
        p = p[:match.start()] + replacer + p[match.end():]
    if (
        not (p[0] == "'" and p[-1] == "'") 
        and p[0] in constants.ALL_QUOTES 
        and len(re.findall(constants.ANY_DOUBLE_QUOTE_REGEX, p[1:])) % 2 == 0
        ):
        p += '"'
    return p.strip()

def prep_document_for_quote_detection(t: str, para_char: str="\n") -> str:
    """
    Splits text into paragraphs (on para_char), runs prep_text_for_quote_detection on all paragraphs, then reassembles with para_char.

    Input:
        t (str) - document to prep for quote detection
        para_char (str) - paragraph boundary in t

    Output:
        document prepped for quote detection
    """
    return para_char.join([prep_text_for_quote_detection(t) for t in t.split(para_char) if t])