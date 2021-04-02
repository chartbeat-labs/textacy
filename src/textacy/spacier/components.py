"""
:mod:`textacy.spacier.components`: Custom components to add to a spaCy language pipeline.
"""
# TODO: figure out why this breaks the code...
# from __future__ import annotations

import inspect
import logging
from typing import Collection, Optional, Union

import spacy
from spacy.tokens import Doc

from .. import text_stats

LOGGER = logging.getLogger(__name__)

_TS_ATTRS = tuple(
    name
    for name, _ in inspect.getmembers(
        text_stats.api.TextStats, lambda member: not(inspect.isroutine(member))
    )
    if not name.startswith("_")
)


class TextStatsComponent:
    """
    A custom component to be added to a spaCy language pipeline that computes
    one, some, or all text stats for a parsed doc and sets the values
    as custom attributes on a :class:`spacy.tokens.Doc`.

    Add the component to a pipeline, *after* the parser (as well as any
    subsequent components that modify the tokens/sentences of the doc)::

        >>> en = spacy.load("en_core_web_sm")
        >>> en.add_pipe("textacy_text_stats", after="parser")

    Process a text with the pipeline and access the custom attributes via
    spaCy's underscore syntax::

        >>> doc = en(u"This is a test test someverylongword.")
        >>> doc._.n_words
        6
        >>> doc._.flesch_reading_ease
        73.84500000000001

    Specify which attributes of the :class:`textacy.text_stats.TextStats()`
    to add to processed documents::

        >>> en = spacy.load("en_core_web_sm")
        >>> en.add_pipe("textacy_text_stats", last=True, config={"attrs": "n_words"})
        >>> doc = en(u"This is a test test someverylongword.")
        >>> doc._.n_words
        6
        >>> doc._.flesch_reading_ease
        AttributeError: [E046] Can't retrieve unregistered extension attribute 'flesch_reading_ease'. Did you forget to call the `set_extension` method?

    Args:
        attrs (str or Iterable[str] or None): If str, a single text stat
            to compute and set on a :obj:`Doc`. If Iterable[str], multiple
            text stats. If None, *all* text stats are computed and set as extensions.

    Attributes:
        name (str): Default name of this component in a spaCy language pipeline,
            used to get and modify the component via various ``spacy.Language``
            methods, e.g. https://spacy.io/api/language#get_pipe.

    See Also:
        :class:`textacy.text_stats.TextStats`
    """

    name = "textacy_text_stats"

    def __init__(self, attrs: Optional[Union[str, Collection[str]]] = None):
        if attrs is None:
            self.attrs = _TS_ATTRS
        elif isinstance(attrs, (str, bytes)):
            self.attrs = (attrs,)
        else:
            self.attrs = tuple(attrs)
        for attr in self.attrs:
            # TODO: see if there's a better way to handle this
            # that doesn't involve clobbering existing property extensions
            Doc.set_extension(attr, default=None, force=True)
            LOGGER.debug('"%s" custom attribute added to `spacy.tokens.Doc`')

    def __call__(self, doc: Doc) -> Doc:
        ts = text_stats.TextStats(doc)
        for attr in self.attrs:
            try:
                doc._.set(attr, getattr(ts, attr))
            except AttributeError:
                LOGGER.exception(
                    "`TextStats` class doesn't have '%s' attribute, so it can't "
                    "be set on this `Doc`. Check the attrs used to initialize "
                    "the `TextStatsComponent` in this pipeline for errors.",
                    attr,
                )
                raise
        return doc


@spacy.language.Language.factory("textacy_text_stats", default_config={"attrs": None})
def text_stats_component(nlp, name, attrs: Optional[Union[str, Collection[str]]]):
    return TextStatsComponent(attrs=attrs)
