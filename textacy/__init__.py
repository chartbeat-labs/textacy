from __future__ import absolute_import

import logging

# subpackages
from textacy import corpora
from textacy import fileio
from textacy import representations
# top-level modules
from textacy import compat, data, math_utils, regexes_etc
from textacy import lexicon_methods, preprocess, text_stats, text_utils
from textacy import spacy_utils
from textacy import extract
from textacy import export, keyterms#, topic_modeling
from textacy import texts

from textacy.texts import TextDoc, TextCorpus

__version__ = '0.2.0'

logger = logging.getLogger('textacy')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
