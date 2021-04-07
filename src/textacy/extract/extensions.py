from typing import Any, Dict

from spacy.tokens import Doc

from . import acros, basics, keyterms, kwic, matches, triples
from .. import errors


def extract_keyterms(doc: Doc, method: str, **kwargs):
    """
    Extract keyterms from a document using one of several different methods.
    For full detail, see the underlying functions listed below.

    See Also:
        - :func:`textacy.extract.keyterms.scake()`
        - :func:`textacy.extract.keyterms.sgrank()`
        - :func:`textacy.extract.keyterms.textrank()`
        - :func:`textacy.extract.keyterms.yake()`
    """
    if method == "scake":
        return keyterms.scake(doc, **kwargs)
    elif method == "sgrank":
        return keyterms.sgrank(doc, **kwargs)
    elif method == "textrank":
        return keyterms.textrank(doc, **kwargs)
    elif method == "yake":
        return keyterms.yake(doc, **kwargs)
    else:
        raise errors.value_invalid_msg(
            "method", method, {"scake", "sgrank", "textrank", "yake"}
        )


def get_doc_extensions() -> Dict[str, Dict[str, Any]]:
    """
    Get :mod:`textacy.extract` custom property and method doc extensions
    that can be set on or removed from the global :class:`spacy.tokens.Doc`.
    """
    return _DOC_EXTENSIONS


def set_doc_extensions():
    """
    Set :mod:`textacy.extract` custom property and method doc extensions
    on the global :class:`spacy.tokens.Doc`.
    """
    for name, kwargs in get_doc_extensions().items():
        if not Doc.has_extension(name):
            Doc.set_extension(name, **kwargs)


def remove_doc_extensions():
    """
    Remove :mod:`textacy.extract` custom property and method doc extensions
    from the global :class:`spacy.tokens.Doc`.
    """
    for name in get_doc_extensions().keys():
        _ = Doc.remove_extension(name)


_DOC_EXTENSIONS: Dict[str, Dict[str, Any]] = {
    "extract_words": {"method": basics.words},
    "extract_ngrams": {"method": basics.ngrams},
    "extract_entities": {"method": basics.entities},
    "extract_noun_chunks": {"method": basics.noun_chunks},
    "extract_terms": {"method": basics.terms},
    "extract_token_matches": {"method": matches.token_matches},
    "extract_regex_matches": {"method": matches.regex_matches},
    "extract_subject_verb_object_triples": {"method": triples.subject_verb_object_triples},
    "extract_semistructured_statements": {"method": triples.semistructured_statements},
    "extract_direct_quotations": {"method": triples.direct_quotations},
    "extract_acronyms": {"method": acros.acronyms},
    "extract_acronyms_and_definitions": {"method": acros.acronyms_and_definitions},
    "extract_keyword_in_context": {"method": kwic.keyword_in_context},
    "extract_keyterms": {"method": extract_keyterms},
}
