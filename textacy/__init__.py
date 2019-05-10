from __future__ import absolute_import

import logging
import os

from .about import __version__


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

from textacy.cache import load_spacy
from textacy.preprocess import preprocess_text
from textacy.doc import make_spacy_doc, remove_doc_extensions, set_doc_extensions
from textacy.corpus import Corpus
from textacy.text_stats import TextStats
from textacy.tm import TopicModel
from textacy.vsm import Vectorizer

set_doc_extensions()

logger = logging.getLogger("textacy")
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())
