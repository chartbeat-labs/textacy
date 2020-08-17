"""
Functions to extract various elements of interest from documents already parsed
by spaCy, such as n-grams, named entities, subject-verb-object triples, and
acronyms.
"""
import collections
import itertools
import operator
import re
from typing import (
    cast,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from cytoolz import itertoolz
from spacy.parts_of_speech import CONJ, DET, NOUN, VERB
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

from . import constants
from . import errors
from . import text_utils
from . import utils
from .spacier import utils as spacy_utils


def words(
    doc: Union[Doc, Span],
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: Optional[Union[str, Set[str]]] = None,
    exclude_pos: Optional[Union[str, Set[str]]] = None,
    min_freq: int = 1,
) -> Iterable[Token]:
    """
    Extract an ordered sequence of words from a document processed by spaCy,
    optionally filtering words by part-of-speech tag and frequency.

    Args:
        doc
        filter_stops: If True, remove stop words from word list.
        filter_punct: If True, remove punctuation from word list.
        filter_nums: If True, remove number-like words (e.g. 10, "ten")
            from word list.
        include_pos: Remove words whose part-of-speech tag IS NOT in the specified tags.
        exclude_pos: Remove words whose part-of-speech tag IS in the specified tags.
        min_freq: Remove words that occur in ``doc`` fewer than ``min_freq`` times.

    Yields:
        Next token from ``doc`` passing specified filters in order of appearance
        in the document.

    Raises:
        TypeError: if ``include_pos`` or ``exclude_pos`` is not a str, a set of str,
            or a falsy value

    Note:
        Filtering by part-of-speech tag uses the universal POS tag set; for details,
        check spaCy's docs: https://spacy.io/api/annotation#pos-tagging
    """
    words_: Iterable[Token] = (w for w in doc if not w.is_space)
    if filter_stops is True:
        words_ = (w for w in words_ if not w.is_stop)
    if filter_punct is True:
        words_ = (w for w in words_ if not w.is_punct)
    if filter_nums is True:
        words_ = (w for w in words_ if not w.like_num)
    if include_pos:
        include_pos = cast(Set[str], utils.to_collection(include_pos, str, set))
        include_pos = {pos.upper() for pos in include_pos}
        words_ = (w for w in words_ if w.pos_ in include_pos)
    if exclude_pos:
        exclude_pos = cast(Set[str], utils.to_collection(exclude_pos, str, set))
        exclude_pos = {pos.upper() for pos in exclude_pos}
        words_ = (w for w in words_ if w.pos_ not in exclude_pos)
    if min_freq > 1:
        words_ = list(words_)
        freqs = itertoolz.frequencies(w.lower_ for w in words_)
        words_ = (w for w in words_ if freqs[w.lower_] >= min_freq)

    for word in words_:
        yield word


def ngrams(
    doc: Union[Doc, Span],
    n: int,
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: Optional[Union[str, Set[str]]] = None,
    exclude_pos: Optional[Union[str, Set[str]]] = None,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of n-grams (``n`` consecutive words) from a
    spacy-parsed doc, optionally filtering n-grams by the types and
    parts-of-speech of the constituent words.

    Args:
        doc
        n: Number of tokens per n-gram; 2 => bigrams, 3 => trigrams, etc.
        filter_stops: If True, remove ngrams that start or end with a stop word
        filter_punct: If True, remove ngrams that contain any punctuation-only tokens
        filter_nums: If True, remove ngrams that contain any numbers
            or number-like tokens (e.g. 10, 'ten')
        include_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE NOT included in this param
        exclude_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE included in this param
        min_freq: Remove ngrams that occur in ``doc`` fewer than ``min_freq`` times

    Yields:
        Next ngram from ``doc`` passing all specified filters, in order of appearance
        in the document

    Raises:
        ValueError: if ``n`` < 1
        TypeError: if ``include_pos`` or ``exclude_pos`` is not a str, a set of str,
            or a falsy value

    Note:
        Filtering by part-of-speech tag uses the universal POS tag set; for details,
        check spaCy's docs: https://spacy.io/api/annotation#pos-tagging
    """
    if n < 1:
        raise ValueError("n must be greater than or equal to 1")

    ngrams_: Iterable[Span] = (doc[i : i + n] for i in range(len(doc) - n + 1))
    ngrams_ = (ngram for ngram in ngrams_ if not any(w.is_space for w in ngram))
    if filter_stops is True:
        ngrams_ = (
            ngram for ngram in ngrams_ if not ngram[0].is_stop and not ngram[-1].is_stop
        )
    if filter_punct is True:
        ngrams_ = (ngram for ngram in ngrams_ if not any(w.is_punct for w in ngram))
    if filter_nums is True:
        ngrams_ = (ngram for ngram in ngrams_ if not any(w.like_num for w in ngram))
    if include_pos:
        include_pos = cast(Set[str], utils.to_collection(include_pos, str, set))
        include_pos = {pos.upper() for pos in include_pos}
        ngrams_ = (
            ngram for ngram in ngrams_ if all(w.pos_ in include_pos for w in ngram)
        )
    if exclude_pos:
        exclude_pos = cast(Set[str], utils.to_collection(exclude_pos, str, set))
        exclude_pos = {pos.upper() for pos in exclude_pos}
        ngrams_ = (
            ngram for ngram in ngrams_ if not any(w.pos_ in exclude_pos for w in ngram)
        )
    if min_freq > 1:
        ngrams_ = list(ngrams_)
        freqs = itertoolz.frequencies(ngram.lower_ for ngram in ngrams_)
        ngrams_ = (ngram for ngram in ngrams_ if freqs[ngram.lower_] >= min_freq)

    for ngram in ngrams_:
        yield ngram


def entities(
    doc: Doc,
    *,
    include_types: Optional[Union[str, Set[str]]] = None,
    exclude_types: Optional[Union[str, Set[str]]] = None,
    drop_determiners: bool = True,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of named entities (PERSON, ORG, LOC, etc.) from
    a ``Doc``, optionally filtering by entity types and frequencies.

    Args:
        doc
        include_types: Remove entities whose type IS NOT
            in this param; if "NUMERIC", all numeric entity types ("DATE",
            "MONEY", "ORDINAL", etc.) are included
        exclude_types: Remove entities whose type IS
            in this param; if "NUMERIC", all numeric entity types ("DATE",
            "MONEY", "ORDINAL", etc.) are excluded
        drop_determiners: Remove leading determiners (e.g. "the")
            from entities (e.g. "the United States" => "United States").

            .. note:: Entities from which a leading determiner has been removed
               are, effectively, *new* entities, and not saved to the ``Doc``
               from which they came. This is irritating but unavoidable, since
               this function is not meant to have side-effects on document state.
               If you're only using the text of the returned spans, this is no
               big deal, but watch out if you're counting on determiner-less
               entities associated with the doc downstream.

        min_freq: Remove entities that occur in ``doc`` fewer
            than ``min_freq`` times

    Yields:
        Next entity from ``doc`` passing all specified filters in order of appearance
        in the document

    Raises:
        TypeError: if ``include_types`` or ``exclude_types`` is not a str, a set of
            str, or a falsy value
    """
    ents = doc.ents
    # HACK: spacy's models have been erroneously tagging whitespace as entities
    # https://github.com/explosion/spaCy/commit/1e6725e9b734862e61081a916baf440697b9971e
    ents = (ent for ent in ents if not ent.text.isspace())
    include_types = _parse_ent_types(include_types, "include")
    exclude_types = _parse_ent_types(exclude_types, "exclude")
    if include_types:
        if isinstance(include_types, str):
            ents = (ent for ent in ents if ent.label_ == include_types)
        elif isinstance(include_types, (set, frozenset, list, tuple)):
            ents = (ent for ent in ents if ent.label_ in include_types)
    if exclude_types:
        if isinstance(exclude_types, str):
            ents = (ent for ent in ents if ent.label_ != exclude_types)
        elif isinstance(exclude_types, (set, frozenset, list, tuple)):
            ents = (ent for ent in ents if ent.label_ not in exclude_types)
    if drop_determiners is True:
        ents = (
            ent
            if ent[0].pos != DET
            else Span(
                ent.doc, ent.start + 1, ent.end, label=ent.label, vector=ent.vector
            )
            for ent in ents
        )
    if min_freq > 1:
        ents = list(ents)
        freqs = itertoolz.frequencies(ent.lower_ for ent in ents)
        ents = (ent for ent in ents if freqs[ent.lower_] >= min_freq)

    for ent in ents:
        yield ent


def _parse_ent_types(
    ent_types: Optional[Union[str, Collection[str]]], which: str,
) -> Optional[Union[str, Set[str]]]:
    if not ent_types:
        return None
    elif isinstance(ent_types, str):
        ent_types = ent_types.upper()
        # replace the shorthand numeric case by its corresponding constant
        if ent_types == "NUMERIC":
            return constants.NUMERIC_ENT_TYPES
        else:
            return ent_types
    elif isinstance(ent_types, (set, frozenset, list, tuple)):
        ent_types = {ent_type.upper() for ent_type in ent_types}
        # again, replace the shorthand numeric case by its corresponding constant
        # and include it in the set in case other types are specified
        if any(ent_type == "NUMERIC" for ent_type in ent_types):
            return ent_types.union(constants.NUMERIC_ENT_TYPES)
        else:
            return ent_types
    else:
        allowed_types = (None, str, set, frozenset, list, tuple)
        raise TypeError(
            errors.type_invalid_msg(
                f"{which}_types", type(ent_types), Optional[Union[str, Collection[str]]]
            )
        )


def noun_chunks(
    doc: Doc, *, drop_determiners: bool = True, min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of noun chunks from a spacy-parsed doc, optionally
    filtering by frequency and dropping leading determiners.

    Args:
        doc
        drop_determiners: Remove leading determiners (e.g. "the")
            from phrases (e.g. "the quick brown fox" => "quick brown fox")
        min_freq: Remove chunks that occur in ``doc`` fewer than ``min_freq`` times

    Yields:
        Next noun chunk from ``doc`` in order of appearance in the document
    """
    ncs = doc.noun_chunks
    if drop_determiners is True:
        ncs = (nc if nc[0].pos != DET else nc[1:] for nc in ncs)
    if min_freq > 1:
        ncs = list(ncs)
        freqs = itertoolz.frequencies(nc.lower_ for nc in ncs)
        ncs = (nc for nc in ncs if freqs[nc.lower_] >= min_freq)

    for nc in ncs:
        yield nc


def pos_regex_matches(doc: Union[Doc, Span], pattern: str) -> Iterable[Span]:
    """
    Extract sequences of consecutive tokens from a spacy-parsed doc whose
    part-of-speech tags match the specified regex pattern.

    Args:
        doc
        pattern: Pattern of consecutive POS tags whose corresponding words
            are to be extracted, inspired by the regex patterns used in NLTK's
            `nltk.chunk.regexp`. Tags are uppercase, from the universal tag set;
            delimited by < and >, which are basically converted to parentheses
            with spaces as needed to correctly extract matching word sequences;
            white space in the input doesn't matter.

            Examples (see ``constants.POS_REGEX_PATTERNS``):

            * noun phrase: r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
            * compound nouns: r'<NOUN>+'
            * verb phrase: r'<VERB>?<ADV>*<VERB>+'
            * prepositional phrase: r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    Yields:
        Next span of consecutive tokens from ``doc`` whose parts-of-speech
        match ``pattern``, in order of appearance

    Warning:
        *DEPRECATED!* For similar but more powerful and performant functionality,
        use :func:`textacy.extract.matches()` instead.
    """
    utils.deprecated(
        "`pos_regex_matches()` has been deprecated! "
        "for similar but more powerful and performant functionality, "
        "use `textacy.extract.matches()` instead.",
        action="once",
    )
    # standardize and transform the regular expression pattern...
    pattern = re.sub(r"\s", "", pattern)
    pattern = re.sub(r"<([A-Z]+)\|([A-Z]+)>", r"( (\1|\2))", pattern)
    pattern = re.sub(r"<([A-Z]+)>", r"( \1)", pattern)

    tags = " " + " ".join(tok.pos_ for tok in doc)

    for m in re.finditer(pattern, tags):
        yield doc[tags[0 : m.start()].count(" ") : tags[0 : m.end()].count(" ")]


def matches(
    doc: Doc,
    patterns: Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
    *,
    on_match: Callable = None,
) -> Iterable[Span]:
    """
    Extract ``Span`` s from a ``Doc`` matching one or more patterns
    of per-token attr:value pairs, with optional quantity qualifiers.

    Args:
        doc
        patterns:
            One or multiple patterns to match against ``doc``
            using a :class:`spacy.matcher.Matcher`.

            If List[dict] or List[List[dict]], each pattern is specified
            as attr: value pairs per token, with optional quantity qualifiers:

            * ``[{"POS": "NOUN"}]`` matches singular or plural nouns,
              like "friend" or "enemies"
            * ``[{"POS": "PREP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "?"}, {"POS": "NOUN", "OP": "+"}]``
              matches prepositional phrases, like "in the future" or "from the distant past"
            * ``[{"IS_DIGIT": True}, {"TAG": "NNS"}]`` matches numbered plural nouns,
              like "60 seconds" or "2 beers"
            * ``[{"POS": "PROPN", "OP": "+"}, {}]`` matches proper nouns and
              whatever word follows them, like "Burton DeWilde yaaasss"

            If str or List[str], each pattern is specified as one or more
            per-token patterns separated by whitespace where attribute, value,
            and optional quantity qualifiers are delimited by colons. Note that
            boolean and integer values have special syntax --- "bool(val)" and
            "int(val)", respectively --- and that wildcard tokens still need
            a colon between the (empty) attribute and value strings.

            * ``"POS:NOUN"`` matches singular or plural nouns
            * ``"POS:PREP POS:DET:? POS:ADJ:? POS:NOUN:+"`` matches prepositional phrases
            * ``"IS_DIGIT:bool(True) TAG:NNS"`` matches numbered plural nouns
            * ``"POS:PROPN:+ :"`` matches proper nouns and whatever word follows them

            Also note that these pattern strings don't support spaCy v2.1's
            "extended" pattern syntax; if you need such complex patterns, it's
            probably better to use a List[dict] or List[List[dict]], anyway.

        on_match: Callback function to act on matches.
            Takes the arguments ``matcher``, ``doc``, ``i`` and ``matches``.

    Yields:
        Next matching ``Span`` in ``doc``, in order of appearance

    Raises:
        TypeError
        ValueError

    See Also:
        * https://spacy.io/usage/rule-based-matching
        * https://spacy.io/api/matcher
    """  # noqa: E501
    if isinstance(patterns, str):
        patterns = [_make_pattern_from_string(patterns)]
    elif isinstance(patterns, (list, tuple)):
        if all(isinstance(item, str) for item in patterns):
            patterns = [_make_pattern_from_string(pattern) for pattern in patterns]
        elif all(isinstance(item, dict) for item in patterns):
            patterns = [patterns]
        elif all(isinstance(item, (list, tuple)) for item in patterns):
            pass  # already in the right format!
        else:
            raise TypeError(
                errors.type_invalid_msg(
                    "patterns",
                    type(patterns),
                    Union[
                        str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]
                    ],
                )
            )
    else:
        raise TypeError(
            errors.type_invalid_msg(
                "patterns",
                type(patterns),
                Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
            )
        )
    matcher = Matcher(doc.vocab)
    matcher.add("match", on_match, *patterns)
    for _, start, end in matcher(doc):
        yield doc[start:end]


def _make_pattern_from_string(patstr: str) -> List[Dict[str, str]]:
    pattern = []
    for tokpatstr in constants.RE_MATCHER_TOKPAT_DELIM.split(patstr):
        parts = tokpatstr.split(":")
        if 2 <= len(parts) <= 3:
            attr = parts[0]
            attr_val = parts[1]
            if attr and attr_val:
                # handle special bool and int attribute values
                special_val = constants.RE_MATCHER_SPECIAL_VAL.match(attr_val)
                if special_val:
                    attr_val = eval(special_val.group(0))
                tokpat = {attr: attr_val}
            # handle wildcard tokens
            else:
                tokpat = {}
            # handle quantifier ops
            try:
                op_val = parts[2]
                if op_val in constants.MATCHER_VALID_OPS:
                    tokpat["OP"] = op_val
                else:
                    raise ValueError(
                        errors.value_invalid_msg(
                            "op", op_val, constants.MATCHER_VALID_OPS
                        )
                    )
            except IndexError:
                pass
            pattern.append(tokpat)
        else:
            raise ValueError(
                f"pattern string '{patstr}' is invalid; "
                "each element in a pattern string must contain an attribute, "
                "a corresponding value, and an optional quantity qualifier, "
                "delimited by colons, like attr:value:op"
            )
    return pattern


def subject_verb_object_triples(
    doc: Union[Doc, Span],
) -> Iterable[Tuple[Span, Span, Span]]:
    """
    Extract an ordered sequence of subject-verb-object (SVO) triples from a
    spacy-parsed doc. Note that this only works for SVO languages.

    Args:
        doc

    Yields:
        Next 3-tuple of spans from ``doc`` representing a (subject, verb, object) triple,
        in order of appearance
    """
    # TODO: What to do about questions, where it may be VSO instead of SVO?
    # TODO: What about non-adjacent verb negations?
    # TODO: What about object (noun) negations?
    if isinstance(doc, Span):
        sents = [doc]
    else:  # spacy.Doc
        sents = doc.sents

    for sent in sents:
        start_i = sent[0].i

        verbs = spacy_utils.get_main_verbs_of_sent(sent)
        for verb in verbs:
            subjs = spacy_utils.get_subjects_of_verb(verb)
            if not subjs:
                continue
            objs = spacy_utils.get_objects_of_verb(verb)
            if not objs:
                continue

            # add adjacent auxiliaries to verbs, for context
            # and add compounds to compound nouns
            verb_span = spacy_utils.get_span_for_verb_auxiliaries(verb)
            verb = sent[verb_span[0] - start_i : verb_span[1] - start_i + 1]
            for subj in subjs:
                subj = sent[
                    spacy_utils.get_span_for_compound_noun(subj)[0]
                    - start_i : subj.i
                    - start_i
                    + 1
                ]
                for obj in objs:
                    if obj.pos == NOUN:
                        span = spacy_utils.get_span_for_compound_noun(obj)
                    elif obj.pos == VERB:
                        span = spacy_utils.get_span_for_verb_auxiliaries(obj)
                    else:
                        span = (obj.i, obj.i)
                    obj = sent[span[0] - start_i : span[1] - start_i + 1]

                    yield (subj, verb, obj)


def acronyms_and_definitions(
    doc: Union[Doc, Span], known_acro_defs: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Extract a collection of acronyms and their most likely definitions, if available,
    from a spacy-parsed doc. If multiple definitions are found for a given acronym,
    only the most frequently occurring definition is returned.

    Args:
        doc
        known_acro_defs: If certain acronym/definition pairs
            are known, pass them in as {acronym (str): definition (str)};
            algorithm will not attempt to find new definitions

    Returns:
        Unique acronyms (keys) with matched definitions (values)

    References:
        Taghva, Kazem, and Jeff Gilbreth. "Recognizing acronyms and their definitions."
        International Journal on Document Analysis and Recognition 1.4 (1999): 191-198.
    """
    # process function arguments
    acro_defs: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    if not known_acro_defs:
        known_acronyms = set()
    else:
        for acro, def_ in known_acro_defs.items():
            acro_defs[acro] = [(def_, 1.0)]
        known_acronyms = set(acro_defs.keys())

    if isinstance(doc, Span):
        sents = [doc]
    else:  # spacy.Doc
        sents = doc.sents

    # iterate over sentences and their tokens
    for sent in sents:
        max_ind = len(sent) - 1

        for i, token in enumerate(sent):

            token_ = token.text
            if token_ in known_acronyms or text_utils.is_acronym(token_) is False:
                continue

            # define definition search window(s)
            window_size = min(2 * len(token_), len(token_) + 5)
            windows = [
                sent[max(i - window_size, 0) : i],
                sent[min(i + 1, max_ind) : min(i + window_size + 1, max_ind)],
            ]
            # if candidate inside (X) or -X-, only look in pre-window
            if 0 < i < max_ind:
                adjacent_tokens = sent[i - 1].text + sent[i + 1].text
                if adjacent_tokens in {"()", "--", "––"}:
                    windows.pop()

            # iterate over possible windows
            # filtering for valid definition strings
            for window in windows:
                window_ = window.text
                # window text can't be all uppercase
                if window_.isupper():
                    continue
                # window can't contain separating punctuation
                if "!" in window_ or "?" in window_ or ":" in window_ or ";" in window_:
                    continue
                # acronym definition can't contain itself: no ouroboros!
                if token_ in window_:
                    continue
                # window must contain at least one character used in acronym
                if not any(char in window_ for char in token_):
                    continue
                definition, confidence = _get_acronym_definition(
                    token_, window, threshold=0.8
                )
                if definition:
                    acro_defs[token_].append((definition, confidence))

            if not acro_defs.get(token_):
                acro_defs[token_].append(("", 0.0))

    # vote by confidence score in the case of multiple definitions
    for acro, defs in acro_defs.items():
        if len(defs) == 1:
            acro_defs[acro] = defs[0][0]
        else:
            acro_defs[acro] = sorted(defs, key=operator.itemgetter(1), reverse=True)[0][
                0
            ]

    return dict(acro_defs)


def _get_acronym_definition(
    acronym: str, window: Span, threshold: float = 0.8,
) -> Tuple[str, float]:
    """
    Identify most likely definition for an acronym given a list of tokens.

    Args:
        acronym: acronym for which definition is sought
        window: a span of tokens from which definition extraction will be attempted
        threshold: minimum "confidence" in definition required for acceptance;
            valid values in [0.0, 1.0]; higher value => stricter threshold

    Returns:
        Most likely definition for given acronym ('' if none found),
        along with the confidence assigned to it

    References:
        Taghva, Kazem, and Jeff Gilbreth. "Recognizing acronyms and their definitions."
        International Journal on Document Analysis and Recognition 1.4 (1999): 191-198.
    """

    def build_lcs_matrix(X, Y):
        m = len(X)
        n = len(Y)
        b = np.zeros((m, n), dtype=int)
        c = np.zeros((m, n), dtype=int)
        for i in range(0, m):
            for j in range(0, n):
                if X[i] == Y[j]:
                    c[i, j] = c[i - 1, j - 1] + 1
                    b[i, j] = 1
                elif c[i - 1, j] >= c[i, j - 1]:
                    c[i, j] = c[i - 1, j]
                else:
                    c[i, j] = c[i, j - 1]
        return c, b

    def parse_lcs_matrix(b, start_i, start_j, lcs_length, stack, vectors):
        m = b.shape[0]
        n = b.shape[1]
        for i in range(start_i, m):
            for j in range(start_j, n):
                if b[i, j] == 1:
                    s = (i, j)
                    stack.append(s)
                    if lcs_length == 1:
                        vec = [np.NaN] * n
                        for k, l in stack:
                            vec[l] = k
                        vectors.append(vec)
                    else:
                        parse_lcs_matrix(b, i + 1, j + 1, lcs_length - 1, stack, vectors)
                    stack = []
        return vectors

    def vector_values(v, types):
        vv = {}
        first = v.index(int(np.nanmin(v)))
        last = v.index(int(np.nanmax(v)))
        vv["size"] = (last - first) + 1
        vv["distance"] = len(v) - last
        vv["stop_count"] = 0
        vv["misses"] = 0
        for i in range(first, last + 1):
            if v[i] >= 0 and types[i] == "s":
                vv["stop_count"] += 1
            elif v[i] is None and types[i] not in ["s", "h"]:
                vv["misses"] += 1
        return vv

    def compare_vectors(A, B, types):
        vv_A = vector_values(A, types)
        vv_B = vector_values(B, types)
        # no one-letter matches, sorryboutit
        if vv_A["size"] == 1:
            return B
        elif vv_B["size"] == 1:
            return A
        if vv_A["misses"] > vv_B["misses"]:
            return B
        elif vv_A["misses"] < vv_B["misses"]:
            return A
        if vv_A["stop_count"] > vv_B["stop_count"]:
            return B
        if vv_A["stop_count"] < vv_B["stop_count"]:
            return A
        if vv_A["distance"] > vv_B["distance"]:
            return B
        elif vv_A["distance"] < vv_B["distance"]:
            return A
        if vv_A["size"] > vv_B["size"]:
            return B
        elif vv_A["size"] < vv_B["size"]:
            return A
        return A

    # get definition window's leading characters and word types
    def_leads = []
    def_types = []
    for tok in window:
        tok_text = tok.text
        if tok.is_stop:
            def_leads.append(tok_text[0])
            def_types.append("s")
        elif text_utils.is_acronym(tok_text):
            def_leads.append(tok_text[0])
            def_types.append("a")
        elif "-" in tok_text and not tok_text.startswith("-"):
            tok_split = [t[0] for t in tok_text.split("-") if t]
            def_leads.extend(tok_split)
            def_types.extend("H" if i == 0 else "h" for i in range(len(tok_split)))
        else:
            def_leads.append(tok_text[0])
            def_types.append("w")
    def_leads = "".join(def_leads).lower()
    def_types = "".join(def_types)

    # extract alphanumeric characters from acronym
    acr_leads = "".join(c for c in acronym if c.isalnum())
    # handle special cases of '&' and trailing 's'
    acr_leads = acr_leads.replace("&", "a")
    if acr_leads.endswith("s"):
        # bail out if it's only a 2-letter acronym to start with, e.g. 'Is'
        if len(acr_leads) == 2:
            return ("", 0)
        acr_leads = acr_leads[:-1]
    acr_leads = acr_leads.lower()

    c, b = build_lcs_matrix(acr_leads, def_leads)

    # 4.4.1
    lcs_length = c[c.shape[0] - 1, c.shape[1] - 1]
    confidence = lcs_length / len(acronym)
    if confidence < threshold:
        return ("", confidence)

    vecs = parse_lcs_matrix(b, 0, 0, lcs_length, [], [])
    # first letter of acronym must be present
    vecs = [vec for vec in vecs if 0 in vec]
    if not vecs:
        return ("", confidence)

    best_vec = vecs[0]
    for vec in vecs[1:]:
        best_vec = compare_vectors(best_vec, vec, def_types)

    first = best_vec.index(int(np.nanmin(best_vec)))
    last = best_vec.index(int(np.nanmax(best_vec)))

    definition = window[first : last + 1].text
    if len(definition.split()) == 1:
        return ("", confidence)

    return (definition, confidence)


def semistructured_statements(
    doc: Doc,
    entity: str,
    *,
    cue: str = "be",
    ignore_entity_case: bool = True,
    min_n_words: int = 1,
    max_n_words: int = 20,
) -> Tuple[Union[Span, Token], Union[Span, Token], Span]:
    """
    Extract "semi-structured statements" from a spacy-parsed doc, each as a
    (entity, cue, fragment) triple. This is similar to subject-verb-object triples.

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


def direct_quotations(doc: Doc) -> Iterable[Tuple[Span, Token, Span]]:
    """
    Baseline, not-great attempt at direction quotation extraction (no indirect
    or mixed quotations) using rules and patterns. English only.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Yields:
        Next quotation in ``doc`` as a (speaker, reporting verb, quotation) triple

    Notes:
        Loosely inspired by Krestel, Bergler, Witte. "Minding the Source: Automatic
        Tagging of Reported Speech in Newspaper Articles".

    TODO: Better approach would use ML, but needs a training dataset.
    """
    doc_lang = doc.vocab.lang
    if doc_lang != "en":
        raise NotImplementedError("sorry, English-language texts only :(")
    quote_end_punct = {",", ".", "?", "!"}
    quote_indexes = set(
        itertoolz.concat(
            (m.start(), m.end() - 1)
            for m in re.finditer(r"(\".*?\")|(''.*?'')|(``.*?'')", doc.text)
        )
    )
    quote_positions = list(
        itertoolz.partition(2, sorted(tok.i for tok in doc if tok.idx in quote_indexes))
    )
    sents = list(doc.sents)
    sent_positions = [(sent.start, sent.end) for sent in sents]

    for q0, q1 in quote_positions:
        quote = doc[q0 : q1 + 1]

        # we're only looking for direct quotes, not indirect or mixed
        if not any(char in quote_end_punct for char in quote.text[-4:]):
            continue

        # get adjacent sentences
        candidate_sent_indexes = []
        for i, (s0, s1) in enumerate(sent_positions):

            if s0 <= q1 + 1 and s1 > q1:
                candidate_sent_indexes.append(i)
            elif s0 < q0 and s1 >= q0 - 1:
                candidate_sent_indexes.append(i)

        for si in candidate_sent_indexes:
            sent = sents[si]

            # get any reporting verbs
            rvs = [
                tok
                for tok in sent
                if spacy_utils.preserve_case(tok) is False
                and tok.lemma_ in constants.REPORTING_VERBS
                and tok.pos_ == "VERB"
                and not any(oq0 <= tok.i <= oq1 for oq0, oq1 in quote_positions)
            ]

            # get target offset against which to measure distances of NEs
            if rvs:
                if len(rvs) == 1:
                    rv = rvs[0]
                else:
                    min_rv_dist = 1000
                    for rv_candidate in rvs:
                        rv_dist = min(abs(rv_candidate.i - qp) for qp in (q0, q1))
                        if rv_dist < min_rv_dist:
                            rv = rv_candidate
                            min_rv_dist = rv_dist
                        else:
                            break
            else:
                # TODO: do we have no other recourse?!
                continue

            try:
                # rv_subj = _find_subjects(rv)[0]
                rv_subj = spacy_utils.get_subjects_of_verb(rv)[0]
            except IndexError:
                continue
            #         if rv_subj.text in {'he', 'she'}:
            #             for ne in entities(doc, include_types={'PERSON'}):
            #                 if ne.start < rv_subj.i:
            #                     speaker = ne
            #                 else:
            #                     break
            #         else:
            span = spacy_utils.get_span_for_compound_noun(rv_subj)
            speaker = doc[span[0] : span[1] + 1]

            yield (speaker, rv, quote)
            break
