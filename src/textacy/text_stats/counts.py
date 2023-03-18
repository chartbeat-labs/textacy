"""
Annotation Counts
-----------------

:mod:`textacy.text_stats.counts`: Functions for computing the counts
of morphological, part-of-speech, and dependency features on the tokens in a document.
"""
import collections
import collections.abc

from .. import types


def morph(doclike: types.DocLike) -> dict[str, dict[str, int]]:
    """
    Count the number of times each value for a morphological feature appears
    as a token annotation in ``doclike``.

    Args:
        doclike

    Returns:
        Mapping of morphological feature to value counts of occurrence.

    See Also:
        :class:`spacy.tokens.MorphAnalysis`
    """
    morph_counts: collections.abc.Mapping = collections.defaultdict(collections.Counter)
    for tok in doclike:
        for label, val in tok.morph.to_dict().items():
            morph_counts[label][val] += 1
    return {label: dict(val_counts) for label, val_counts in morph_counts.items()}


def tag(doclike: types.DocLike) -> dict[str, int]:
    """
    Count the number of times each fine-grained part-of-speech tag appears
    as a token annotation in ``doclike``.

    Args:
        doclike

    Returns:
        Mapping of part-of-speech tag to count of occurrence.
    """
    return dict(collections.Counter(tok.tag_ for tok in doclike))


def pos(doclike: types.DocLike) -> dict[str, int]:
    """
    Count the number of times each coarsed-grained universal part-of-speech tag appears
    as a token annotation in ``doclike``.

    Args:
        doclike

    Returns:
        Mapping of universal part-of-speech tag to count of occurrence.
    """
    return dict(collections.Counter(tok.pos_ for tok in doclike))


def dep(doclike: types.DocLike) -> dict[str, int]:
    """
    Count the number of times each syntactic dependency relation appears
    as a token annotation in ``doclike``.

    Args:
        doclike

    Returns:
        Mapping of dependency relation to count of occurrence.
    """
    return dict(collections.Counter(tok.dep_ for tok in doclike))
