"""
Morphological Stats
-------------------

:mod:`textacy.text_stats.morph`: TODO
"""
import collections
import functools
from typing import Dict, Set

from cachetools import cached
from cachetools.keys import hashkey
from spacy.morphology import Morphology
from spacy.pipeline import Morphologizer

from .. import cache, constants, types
from ..spacier import utils as sputils


@cached(cache.LRU_CACHE, key=functools.partial(hashkey, "spacy_lang_morph_labels"))
def get_spacy_lang_morph_labels(lang: types.LangLike) -> Set[str]:
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
    spacy_lang = sputils.resolve_langlike(lang)
    if spacy_lang.has_pipe("morphologizer"):
        morphologizer = spacy_lang.get_pipe("morphologizer")
    elif any(isinstance(comp, Morphologizer) for _, comp in spacy_lang.pipeline):
        for _, component in spacy_lang.pipeline:
            if isinstance(component, Morphologizer):
                morphologizer = component
                break
    else:
        return constants.UD_V2_MORPH_LABELS

    return {
        feat_name
        for label in morphologizer.labels
        for feat_name in Morphology.feats_to_dict(label).keys()
    }


def get_morph_label_counts(label: str, doclike: types.DocLike) -> Dict[str, int]:
    """
    For a given morphological feature ``label``, count the number of times each
    value appears as a token annotation in ``doclike``.

    Args:
        label
        doclike

    Returns:
        Mapping of morphological label to the number of times its corresponding values
        appear in the document.
    """
    return dict(
        collections.Counter(val for tok in doclike for val in tok.morph.get(label))
    )
