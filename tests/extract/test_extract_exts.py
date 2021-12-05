import pytest

import textacy
from textacy import extract


@pytest.mark.parametrize("name", ["extract", "extract.basics", "extract.keyterms"])
def test_set_get_remove_extensions(doc_en, name):
    textacy.set_doc_extensions(name)
    assert all(doc_en.has_extension(n) for n in textacy.get_doc_extensions(name).keys())
    if "." in name:
        assert not all(
            doc_en.has_extension(n) for n in textacy.get_doc_extensions("extract").keys()
        )
    textacy.remove_doc_extensions(name)
    assert not any(
        doc_en.has_extension(n) for n in textacy.get_doc_extensions(name).keys()
    )


@pytest.mark.parametrize(
    "ext_name, kwargs",
    [
        ("extract_words", {}),
        ("extract_ngrams", {"n": 2}),
        ("extract_entities", {}),
        ("extract_noun_chunks", {}),
        ("extract_terms", {"ngs": 2, "ents": True, "ncs": True}),
        ("extract_token_matches", {"patterns": [{"POS": "NOUN"}]}),
        ("extract_regex_matches", {"pattern": "[Mm]any"}),
        ("extract_subject_verb_object_triples", {}),
        ("extract_semistructured_statements", {"entity": "Macondo", "cue": "is"}),
        ("extract_direct_quotations", {}),
        ("extract_acronyms", {}),
        ("extract_keyword_in_context", {"keyword": "Macondo"}),
    ],
)
def test_extensions_match(doc_en, ext_name, kwargs):
    textacy.set_doc_extensions("extract")
    ext = getattr(doc_en._, ext_name)
    func = textacy.get_doc_extensions("extract")[ext_name]["method"]
    ext_val = ext(**kwargs)
    func_val = func(doc_en, **kwargs)
    if isinstance(func_val, dict):
        assert ext_val == func_val
    else:
        assert list(ext_val) == list(func_val)
    textacy.remove_doc_extensions("extract")
