from __future__ import absolute_import

import logging
import os

__version__ = '0.2.4'
__data_dir__ = os.path.join(os.path.dirname(__file__), 'resources')

# subpackages
from textacy import corpora
from textacy import fileio
from textacy import representations
from textacy import tm
from textacy import viz
# top-level modules
from textacy import compat, data, math_utils, regexes_etc
from textacy import lexicon_methods, preprocess, text_stats, text_utils
from textacy import spacy_utils
from textacy import extract
from textacy import distance, export, keyterms
from textacy import texts

from textacy.data import load_spacy
from textacy.preprocess import preprocess_text
from textacy.texts import TextDoc, TextCorpus

logger = logging.getLogger('textacy')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
