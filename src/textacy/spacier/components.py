"""
:mod:`textacy.spacier.components`: Custom components to add to a spaCy language pipeline.
"""
# TODO: figure out why this breaks the code...
# from __future__ import annotations

import inspect
import logging
from typing import Collection, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc

# TODO: should we hide this import in, e.g. the component init?
from .. import text_stats

LOGGER = logging.getLogger(__name__)


class TextStatsComponent:
    """
    A custom component to be added to a spaCy language pipeline that computes
    one, some, or all text stats for a parsed doc and sets the values
    as custom attributes on a :class:`spacy.tokens.Doc`.

    Add the component to a pipeline, *after* the parser and any subsequent components
    that modify the tokens/sentences of the doc (to be safe, just put it last)::

        >>> en = spacy.load("en_core_web_sm")
        >>> en.add_pipe("textacy_text_stats", last=True)

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
        attr: If str, a single text stat to compute and set on a :obj:`Doc`;
            if Iterable[str], set multiple text stats; if None, *all* text stats
            are computed and set as extensions.

    See Also:
        :class:`textacy.text_stats.TextStats`
    """

    def __init__(self, attrs: Optional[Union[str, Collection[str]]] = None):
        self._set_attrs(attrs)
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

    def _set_attrs(self, attrs: Optional[Union[str, Collection[str]]]):
        if attrs is None:
            self.attrs = tuple(
                name
                for name, _ in inspect.getmembers(
                    text_stats.TextStats, lambda memb: not(inspect.isroutine(memb))
                )
                if not name.startswith("_")
            )
        elif isinstance(attrs, str):
            self.attrs = (attrs,)
        else:
            self.attrs = tuple(attrs)


@Language.factory(
    "textacy_text_stats",
    default_config={"attrs": None},
    retokenizes=False,
)
def create_text_stats_component(nlp, name, attrs: Optional[Union[str, Collection[str]]]):
    return TextStatsComponent(attrs=attrs)
