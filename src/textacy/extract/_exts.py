# mypy: ignore-errors
"""
TODO
"""
from __future__ import annotations

from spacy.tokens import Doc

from .. import errors, types
from ..spacier.extensions import doc_extensions_registry
from . import acros, bags, basics, keyterms, kwic, matches, triples


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
        raise ValueError(
            errors.value_invalid_msg(
                "method", method, {"scake", "sgrank", "textrank", "yake"}
            )
        )


@doc_extensions_registry.register("extract.acros")
def _get_doc_extensions_extract_acros() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "extract_acronyms": {"method": acros.acronyms},
        "extract_acronyms_and_definitions": {"method": acros.acronyms_and_definitions},
    }


@doc_extensions_registry.register("extract.bags")
def _get_doc_extensions_extract_bags() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "to_bag_of_words": {"method": bags.to_bag_of_words},
        "to_bag_of_terms": {"method": bags.to_bag_of_terms},
    }


@doc_extensions_registry.register("extract.basics")
def _get_doc_extensions_extract_basics() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "extract_words": {"method": basics.words},
        "extract_ngrams": {"method": basics.ngrams},
        "extract_entities": {"method": basics.entities},
        "extract_noun_chunks": {"method": basics.noun_chunks},
        "extract_terms": {"method": basics.terms},
    }


@doc_extensions_registry.register("extract.kwic")
def _get_doc_extensions_extract_kwic() -> dict[str, dict[str, types.DocExtFunc]]:
    return {"extract_keyword_in_context": {"method": kwic.keyword_in_context}}


@doc_extensions_registry.register("extract.matches")
def _get_doc_extensions_extract_matches() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "extract_token_matches": {"method": matches.token_matches},
        "extract_regex_matches": {"method": matches.regex_matches},
    }


@doc_extensions_registry.register("extract.triples")
def _get_doc_extensions_extract_triples() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "extract_subject_verb_object_triples": {
            "method": triples.subject_verb_object_triples
        },
        "extract_semistructured_statements": {
            "method": triples.semistructured_statements
        },
        "extract_direct_quotations": {"method": triples.direct_quotations},
    }


@doc_extensions_registry.register("extract.keyterms")
def _get_doc_extensions_extract_keyterms() -> dict[str, dict[str, types.DocExtFunc]]:
    return {"extract_keyterms": {"method": extract_keyterms}}


@doc_extensions_registry.register("extract")
def _get_doc_extensions_extract() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        **_get_doc_extensions_extract_acros(),
        **_get_doc_extensions_extract_bags(),
        **_get_doc_extensions_extract_basics(),
        **_get_doc_extensions_extract_kwic(),
        **_get_doc_extensions_extract_matches(),
        **_get_doc_extensions_extract_triples(),
        **_get_doc_extensions_extract_keyterms(),
    }
