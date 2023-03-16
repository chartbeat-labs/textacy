from __future__ import annotations

import collections
import functools
import itertools
import string
from typing import Iterable

from cachetools import cached
from cachetools.keys import hashkey
from spacy.tokens import Doc, Span

from .. import cache, datasets, errors, resources, types


concept_net = resources.ConceptNet()
udhr = datasets.UDHR()


def to_aug_toks(doclike: types.DocLike) -> list[types.AugTok]:
    """
    Transform a spaCy ``Doc`` or ``Span`` into a list of ``AugTok`` objects,
    suitable for use in data augmentation transform functions.
    """
    if not isinstance(doclike, (Doc, Span)):
        raise TypeError(
            errors.type_invalid_msg("spacy_obj", type(doclike), types.DocLike)
        )
    lang = doclike.vocab.lang
    toks_syns: Iterable[list[str]]
    if concept_net.filepath is None or lang not in concept_net.synonyms:
        toks_syns = ([] for _ in doclike)
    else:
        toks_syns = (
            concept_net.get_synonyms(tok.text, lang=lang, sense=tok.pos_)
            if not (tok.is_punct or tok.is_space)
            else []
            for tok in doclike
        )
    return [
        types.AugTok(
            text=tok.text,
            ws=tok.whitespace_,
            pos=tok.pos_,
            is_word=(not (tok.is_punct or tok.is_space)),
            syns=syns,
        )
        for tok, syns in zip(doclike, toks_syns)
    ]


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "char_weights"))
def get_char_weights(lang: str) -> list[tuple[str, int]]:
    """
    Get lang-specific character weights for use in certain data augmentation transforms,
    based on texts in :class:`textacy.datasets.UDHR`.

    Args:
        lang: Standard two-letter language code.

    Returns:
        Collection of (character, weight) pairs, based on the distribution of characters
        found in the source text.
    """
    try:
        char_weights = list(
            collections.Counter(
                char
                for text in udhr.texts(lang=lang)
                for char in text
                if char.isalnum()
            ).items()
        )
    except ValueError:
        char_weights = list(
            zip(string.ascii_lowercase + string.digits, itertools.repeat(1))
        )
    return char_weights
