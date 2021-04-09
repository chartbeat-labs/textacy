from .acros import acronyms, acronyms_and_definitions
from .basics import words, ngrams, entities, noun_chunks, terms
from .kwic import keyword_in_context
from .matches import token_matches, regex_matches
from .triples import (
    direct_quotations, semistructured_statements, subject_verb_object_triples
)
from .utils import terms_to_strings, clean_term_strings, aggregate_term_variants

from .extensions import get_doc_extensions, remove_doc_extensions, set_doc_extensions

set_doc_extensions()
