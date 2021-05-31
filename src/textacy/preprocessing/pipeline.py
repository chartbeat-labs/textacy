"""
Pipeline
--------

:mod:`textacy.preprocessing.pipeline`: Basic functionality for composing multiple
preprocessing steps into a single callable pipeline.
"""
from typing import Callable

from cytoolz import functoolz


def make_pipeline(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """
    Make a callable pipeline that takes a text as input, passes it through one or more
    functions in sequential order, then outputs a single (preprocessed) text string.

    This function is intended as a lightweight convenience for users, allowing them
    to flexibly specify which (and in which order) preprocessing functions are to be
    applied to raw texts, then treating the whole thing as a single callable.

    .. code-block:: pycon

        >>> from textacy import preprocessing
        >>> preproc = preprocessing.make_pipeline(
        ...     preprocessing.replace.hashtags,
        ...     preprocessing.replace.user_handles,
        ...     preprocessing.replace.emojis,
        ... )
        >>> preproc("@spacy_io is OSS for industrial-strength NLP in Python developed by @explosion_ai 💥")
        '_USER_ is OSS for industrial-strength NLP in Python developed by _USER_ _EMOJI_'
        >>> preproc("hacking with my buddy Isaac Mewton 🥰 #PawProgramming")
        'hacking with my buddy Isaac Mewton _EMOJI_ _TAG_'

    To specify arguments for individual preprocessors, use :func:`functools.partial`:

    .. code-block:: pycon

        >>> from functools import partial
        >>> preproc = preprocessing.make_pipeline(
        ...     partial(preprocessing.remove.punctuation, only=[".", "?", "!"]),
        ...     partial(preprocessing.replace.user_handles, repl="TAG"),
        ... )
        >>> preproc("hey, @bjdewilde! when's the next release of textacy?")
        "hey, TAG  when's the next release of textacy "

    Args:
        *funcs

    Returns:
        Pipeline composed of ``*funcs`` that applies each in sequential order.
    """
    return functoolz.compose_left(*funcs)
