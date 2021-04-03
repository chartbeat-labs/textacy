from spacy.tokens import Doc

from . import acros, basics, keyterms, matches, triples


DOC_EXTENSIONS = {
    "extract_words": {"method": basics.words},
    "extract_ngrams": {"method": basics.ngrams},
    "extract_entities": {"method": basics.entities},
    "extract_noun_chunks": {"method": basics.noun_chunks},
    "extract_token_matches": {"method": matches.token_matches},
    "extract_regex_matches": {"method": matches.regex_matches},
    "extract_subject_verb_object_triples": {"method": triples.subject_verb_object_triples},
    "extract_semistructured_statements": {"method": triples.semistructured_statements},
    "extract_direct_quotations": {"method": triples.direct_quotations},
    "extract_acronyms": {"method": acros.acronyms},
    "extract_acronyms_and_definitions": {"method": acros.acronyms_and_definitions},
}


# TODO: update/add funcs to make this convenient via doc extension
#    kwic.keyword_in_context
#    keyterms.textrank
#    keyterms.yake
#    keyterms.scake
#    keyterms.sgrank

def set_doc_extensions():
    """
    Set :mod:`textacy.extract` custom property and method doc extensions
    on the global :class:`spacy.tokens.Doc`.
    """
    for name, kwargs in DOC_EXTENSIONS.items():
        if not Doc.has_extension(name):
            Doc.set_extension(name, **kwargs)


def remove_doc_extensions():
    """
    Remove :mod:`textacy.extract` custom property and method doc extensions
    from the global :class:`spacy.tokens.Doc`.
    """
    for name in DOC_EXTENSIONS.keys():
        _ = Doc.remove_extension(name)
