# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from . import utils

LOGGER = logging.getLogger(__name__)

utils.deprecated(
    "The `spacy_pipelines` module is deprecated and will be removed in v0.7.0."
    "Use the `textacy.spacier` subpackage instead.",
    action="once",
)


def _merge_entities(doc):
    """
    Merge named entities *in-place* within parent ``doc`` so that each becomes
    a single token.

    Args:
        doc (``spacy.doc``)
    """
    for ent in doc.ents:
        try:
            ent.merge(ent.root.tag_, ent.text, ent.root.ent_type_)
        except IndexError as e:
            LOGGER.exception('Unable to merge entity "%s"; skipping...', ent.text)


def merged_entities_pipeline(lang):
    """
    Get spaCy's standard language pipeline — tagger, parser, matcher, entity —
    with an additional step in which entities are merged into single tokens.

    Args:
        lang (``spacy.Language``)

    Returns:
        Tuple[Callable]: Each callable modifies ``SpacyDoc`` objects in-place.
    """
    return (lang.tagger, lang.parser, lang.matcher, lang.entity, _merge_entities)
