"""
Pipeline Components
-------------------

:mod:`textacy.text_stats.components`: Custom components to add to a spaCy language pipeline.
"""
import logging

from spacy.language import Language
from spacy.tokens import Doc

from . import api

LOGGER = logging.getLogger(__name__)


@Language.factory(
    "textacy_text_stats",
    retokenizes=False,
    # assigns=["doc.text_stats"],
    # requires=["token.morph"],
)
def text_stats_component(nlp: Language, name: str):
    return TextStatsComponent()


class TextStatsComponent:
    """
    A custom component to be added to a spaCy Language pipeline
    that assigns a :class:`textacy.text_stats.TextStats` instance
    to the "text_stats" custom ``Doc`` extension.

    Add the component to a pipeline, *after* the parser and any subsequent components
    that modify the tokens/sentences of the doc (to be safe, just put it last)::

        >>> en = spacy.load("en_core_web_sm")
        >>> en.add_pipe("textacy_text_stats", last=True)

    Process a text with the pipeline and access the custom attributes via
    spaCy's underscore syntax::

        >>> doc = en("The year was 2081, and everybody was finally equal.")
        >>> doc._.text_stats
        >>> doc._.text_stats.n_words
        9
        >>> doc._.text_stats.readability("flesch-reading-ease")
        94.30000000000003

    See Also:
        :class:`textacy.text_stats.TextStats`
    """

    def __init__(self):
        if not Doc.has_extension("text_stats"):
            Doc.set_extension("text_stats", default=None)
            LOGGER.debug('"text_stats" custom attribute added to `spacy.tokens.Doc`')

    def __call__(self, doc: Doc) -> Doc:
        doc._.set("text_stats", api.TextStats(doc))
        return doc
