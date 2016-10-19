from __future__ import absolute_import

import logging
import os

__version__ = '0.3.1'
__resources_dir__ = os.path.join(os.path.dirname(__file__), 'resources')

# subpackages
from textacy import corpora
from textacy import fileio
from textacy import tm
from textacy import viz
# top-level modules
from textacy import compat, constants, math_utils, vsm
from textacy import data, preprocess, text_utils
from textacy import lexicon_methods, spacy_utils, text_stats
from textacy import doc
from textacy import corpus
from textacy import extract
from textacy import export, keyterms, network, similarity

from textacy.data import load_spacy
from textacy.preprocess import preprocess_text
from textacy.doc import Doc
from textacy.corpus import Corpus

logger = logging.getLogger('textacy')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
