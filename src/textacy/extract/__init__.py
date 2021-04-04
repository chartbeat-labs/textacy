from . import keyterms
from .acros import acronyms, acronyms_and_definitions
from .basics import words, ngrams, entities, noun_chunks
from .kwic import keyword_in_context
from .matches import token_matches, regex_matches
from .triples import (
    direct_quotations, semistructured_statements, subject_verb_object_triples
)

from .extensions import set_doc_extensions, remove_doc_extensions

set_doc_extensions()
