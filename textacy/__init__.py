from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from textacy.about import __version__
from textacy.constants import DEFAULT_DATA_DIR
from textacy.cache import load_spacy_lang
from textacy.preprocess import preprocess_text
from textacy.doc import make_spacy_doc
from textacy.corpus import Corpus
from textacy.text_stats import TextStats
from textacy.spacier.doc_extensions import set_doc_extensions
# keep these out of the main namespace
# they're somewhat niche, and slow to import bc of heavy dependencies
# from textacy.tm import TopicModel
# from textacy.vsm import Vectorizer

set_doc_extensions()

logger = logging.getLogger("textacy")
# ensure reload() doesn't add another handler
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())
