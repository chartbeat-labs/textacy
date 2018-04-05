"""
Custom components to add to a spaCy language pipeline.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from spacy.tokens import Doc as SpacyDoc

from .. import compat
from .. import text_stats

LOGGER = logging.getLogger(__name__)


class TextStatsComponent(object):
    """
    A custom component to be added to a spaCy language pipeline that computes
    one, some, or all text stats for a parsed doc and sets the values
    as custom attributes on a :obj:`SpacyDoc`.

    Add the component to a pipeline, *after* the parser::

        >>> en = spacy.load('en')
        >>> text_stats_component = TextStatsComponent()
        >>> en.add_pipe(text_stats_component, after='parser')

    Process a text with the pipeline and access the custom attributes via
    spaCy's underscore syntax::

        >>> doc = en(u"This is a test test someverylongword.")
        >>> doc._.n_words
        6
        >>> doc._.flesch_reading_ease
        73.84500000000001

    Specify which attributes of the :class:`textacy.text_stats.TextStats()`
    to add to processed documents::

        >>> en = spacy.load('en')
        >>> text_stats_component = TextStatsComponent(attrs='n_words')
        >>> en.add_pipe(text_stats_component, after='parser')
        >>> doc = en(u"This is a test test someverylongword.")
        >>> doc._.n_words
        6
        >>> doc._.flesch_reading_ease
        AttributeError: [E046] Can't retrieve unregistered extension attribute 'flesch_reading_ease'. Did you forget to call the `set_extension` method?

    Args:
        attrs (str or Iterable[str] or None): If str, a single text stat
            to compute and set on a :obj:`Doc`. If Iterable[str], multiple
            text stats. If None, *all* text stats are computed and set as extensions.

    See Also:
        :class:`textacy.text_stats.TextStats`
    """

    name = 'textacy_text_stats'

    def __init__(self, attrs=None):
        if attrs is None:
            self.attrs = (
                'n_sents', 'n_words', 'n_chars', 'n_syllables',
                'n_unique_words', 'n_long_words',
                'n_monosyllable_words', 'n_polysyllable_words',
                'flesch_kincaid_grade_level', 'flesch_reading_ease',
                'smog_index', 'gunning_fog_index', 'coleman_liau_index',
                'automated_readability_index', 'lix', 'gulpease_index',
                'wiener_sachtextformel',
            )
        elif isinstance(attrs, compat.string_types):
            self.attrs = (attrs,)
        else:
            self.attrs = tuple(attrs)
        for attr in self.attrs:
            SpacyDoc.set_extension(attr, default=None)
            LOGGER.debug('"%s" custom attribute added to `spacy.tokens.Doc`')

    def __call__(self, doc):
        ts = text_stats.TextStats(doc)
        for attr in self.attrs:
            doc._.set(attr, getattr(ts, attr))
        return doc
