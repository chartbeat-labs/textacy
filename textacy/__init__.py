from __future__ import absolute_import

import logging

from textacy.preprocess import preprocess_text
from textacy.texts import TextDoc, TextCorpus
from textacy import data

__all__ = [
    'preprocess_text',
    'TextDoc',
    'TextCorpus'
]

__version__ = '0.1.1'

logger = logging.getLogger('textacy')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
