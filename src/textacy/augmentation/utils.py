import collections
import functools
import itertools
import string
from typing import Iterable, List, Tuple, Union

from cachetools import cached
from cachetools.keys import hashkey
from spacy.tokens import Doc, Span

from .. import cache, datasets, resources


concept_net = resources.ConceptNet()
udhr = datasets.UDHR()

AugTok = collections.namedtuple("AugTok", ["text", "ws", "pos", "is_word", "syns"])
"""tuple: Minimal token data required for data augmentation transforms."""


def to_aug_toks(spacy_obj: Union[Doc, Span]) -> List[AugTok]:
    """
    Transform a spaCy ``Doc`` or ``Span`` into a list of ``AugTok`` objects,
    suitable for use in data augmentation transform functions.
    """
    if not isinstance(spacy_obj, (Doc, Span)):
        raise TypeError(
            "`spacy_obj` must be of type {}, not {}".format((Doc, Span), type(spacy_obj))
        )
    lang = spacy_obj.vocab.lang
    toks_syns: Iterable[List[str]]
    if concept_net.filepath is None or lang not in concept_net.synonyms:
        toks_syns = ([] for _ in spacy_obj)
    else:
        toks_syns = (
            concept_net.get_synonyms(tok.text, lang=lang, sense=tok.pos_)
            if not (tok.is_punct or tok.is_space)
            else []
            for tok in spacy_obj
        )
    return [
        AugTok(
            text=tok.text,
            ws=tok.whitespace_,
            pos=tok.pos_,
            is_word=(not (tok.is_punct or tok.is_space)),
            syns=syns,
        )
        for tok, syns in zip(spacy_obj, toks_syns)
    ]


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "char_weights"))
def get_char_weights(lang: str) -> List[Tuple[str, int]]:
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
                char for text in udhr.texts(lang=lang) for char in text if char.isalnum()
            ).items()
        )
    except ValueError:
        char_weights = list(
            zip(string.ascii_lowercase + string.digits, itertools.repeat(1))
        )
    return char_weights
