"""
Basics
------

:mod:`textacy.extract.basics`: Extract basic components from a document or sentence
via spaCy, with bells and whistles for filtering the results.
"""
from __future__ import annotations

from typing import Collection, Iterable, Optional, Set, Union

from cytoolz import itertoolz
from spacy.parts_of_speech import DET
from spacy.tokens import Doc, Span, Token

from .. import constants
from .. import errors
from .. import utils


def words(
    doclike: Doc | Span,
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: Optional[str | Collection[str]] = None,
    exclude_pos: Optional[str | Collection[str]] = None,
    min_freq: int = 1,
) -> Iterable[Token]:
    """
    Extract an ordered sequence of words from a document processed by spaCy,
    optionally filtering words by part-of-speech tag and frequency.

    Args:
        doclike
        filter_stops: If True, remove stop words from word list.
        filter_punct: If True, remove punctuation from word list.
        filter_nums: If True, remove number-like words (e.g. 10, "ten")
            from word list.
        include_pos: Remove words whose part-of-speech tag IS NOT in the specified tags.
        exclude_pos: Remove words whose part-of-speech tag IS in the specified tags.
        min_freq: Remove words that occur in ``doclike`` fewer than ``min_freq`` times.

    Yields:
        Next token from ``doclike`` passing specified filters in order of appearance
        in the document.

    Raises:
        TypeError: if ``include_pos`` or ``exclude_pos`` is not a str, a set of str,
            or a falsy value

    Note:
        Filtering by part-of-speech tag uses the universal POS tag set; for details,
        check spaCy's docs: https://spacy.io/api/annotation#pos-tagging
    """
    words_: Iterable[Token] = (w for w in doclike if not w.is_space)
    if filter_stops is True:
        words_ = (w for w in words_ if not w.is_stop)
    if filter_punct is True:
        words_ = (w for w in words_ if not w.is_punct)
    if filter_nums is True:
        words_ = (w for w in words_ if not w.like_num)
    if include_pos:
        include_pos = utils.to_collection(include_pos, str, set)
        include_pos = {pos.upper() for pos in include_pos}
        words_ = (w for w in words_ if w.pos_ in include_pos)
    if exclude_pos:
        exclude_pos = utils.to_collection(exclude_pos, str, set)
        exclude_pos = {pos.upper() for pos in exclude_pos}
        words_ = (w for w in words_ if w.pos_ not in exclude_pos)
    if min_freq > 1:
        words_ = list(words_)
        freqs = itertoolz.frequencies(w.lower_ for w in words_)
        words_ = (w for w in words_ if freqs[w.lower_] >= min_freq)

    for word in words_:
        yield word


def ngrams(
    doclike: Doc | Span,
    n: int,
    *,
    filter_stops: bool = True,
    filter_punct: bool = True,
    filter_nums: bool = False,
    include_pos: Optional[str | Collection[str]] = None,
    exclude_pos: Optional[str | Collection[str]] = None,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of n-grams (``n`` consecutive words) from a
    spacy-parsed doc, optionally filtering n-grams by the types and
    parts-of-speech of the constituent words.

    Args:
        doclike
        n: Number of tokens per n-gram; 2 => bigrams, 3 => trigrams, etc.
        filter_stops: If True, remove ngrams that start or end with a stop word
        filter_punct: If True, remove ngrams that contain any punctuation-only tokens
        filter_nums: If True, remove ngrams that contain any numbers
            or number-like tokens (e.g. 10, 'ten')
        include_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE NOT included in this param
        exclude_pos: Remove ngrams if any constituent tokens' part-of-speech tags
            ARE included in this param
        min_freq: Remove ngrams that occur in ``doclike`` fewer than ``min_freq`` times

    Yields:
        Next ngram from ``doclike`` passing all specified filters, in order of appearance
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

    ngrams_: Iterable[Span] = (doclike[i : i + n] for i in range(len(doclike) - n + 1))
    ngrams_ = (ng for ng in ngrams_ if not any(w.is_space for w in ng))
    if filter_stops is True:
        ngrams_ = (ng for ng in ngrams_ if not ng[0].is_stop and not ng[-1].is_stop)
    if filter_punct is True:
        ngrams_ = (ng for ng in ngrams_ if not any(w.is_punct for w in ng))
    if filter_nums is True:
        ngrams_ = (ng for ng in ngrams_ if not any(w.like_num for w in ng))
    if include_pos:
        include_pos = {
            pos.upper()
            for pos in utils.to_collection(include_pos, str, set)
        }
        ngrams_ = (ng for ng in ngrams_ if all(w.pos_ in include_pos for w in ng))
    if exclude_pos:
        exclude_pos = {
            pos.upper()
            for pos in utils.to_collection(exclude_pos, str, set)
        }
        ngrams_ = (ng for ng in ngrams_ if not any(w.pos_ in exclude_pos for w in ng))
    if min_freq > 1:
        ngrams_ = list(ngrams_)
        freqs = itertoolz.frequencies(ng.text.lower() for ng in ngrams_)
        ngrams_ = (ng for ng in ngrams_ if freqs[ng.text.lower()] >= min_freq)

    for ngram in ngrams_:
        yield ngram


def entities(
    doclike: Doc | Span,
    *,
    include_types: Optional[str | Collection[str]] = None,
    exclude_types: Optional[str | Collection[str]] = None,
    drop_determiners: bool = True,
    min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of named entities (PERSON, ORG, LOC, etc.) from
    a ``Doc``, optionally filtering by entity types and frequencies.

    Args:
        doclike
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

        min_freq: Remove entities that occur in ``doclike`` fewer
            than ``min_freq`` times

    Yields:
        Next entity from ``doclike`` passing all specified filters in order of appearance
        in the document

    Raises:
        TypeError: if ``include_types`` or ``exclude_types`` is not a str, a set of
            str, or a falsy value
    """
    ents = doclike.ents
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
        freqs = itertoolz.frequencies(ent.text.lower() for ent in ents)
        ents = (ent for ent in ents if freqs[ent.text.lower()] >= min_freq)

    for ent in ents:
        yield ent


def _parse_ent_types(
    ent_types: Optional[str | Collection[str]], which: str,
) -> Optional[str | Set[str]]:
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
    doclike: Doc | Span, *, drop_determiners: bool = True, min_freq: int = 1,
) -> Iterable[Span]:
    """
    Extract an ordered sequence of noun chunks from a spacy-parsed doc, optionally
    filtering by frequency and dropping leading determiners.

    Args:
        doclike
        drop_determiners: Remove leading determiners (e.g. "the")
            from phrases (e.g. "the quick brown fox" => "quick brown fox")
        min_freq: Remove chunks that occur in ``doclike`` fewer than ``min_freq`` times

    Yields:
        Next noun chunk from ``doclike`` in order of appearance in the document
    """
    ncs = doclike.noun_chunks
    if drop_determiners is True:
        ncs = (nc if nc[0].pos != DET else nc[1:] for nc in ncs)
    if min_freq > 1:
        ncs = list(ncs)
        freqs = itertoolz.frequencies(nc.text.lower() for nc in ncs)
        ncs = (nc for nc in ncs if freqs[nc.text.lower()] >= min_freq)

    for nc in ncs:
        yield nc
