import logging

from textacy._version import __version__
from textacy.constants import DEFAULT_DATA_DIR
from textacy.corpus import Corpus
from textacy.lang_id import identify_lang
from textacy.spacier.core import load_spacy_lang, make_spacy_doc
from textacy.spacier.extensions import (
    get_doc_extensions,
    remove_doc_extensions,
    set_doc_extensions,
)

set_doc_extensions("spacier")

logger = logging.getLogger("textacy")
# ensure reload() doesn't add another handler
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())
